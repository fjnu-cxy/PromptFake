from numpy import dtype
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .clip import load, tokenize
from .adapter_modules import SimpleAdapter, SimpleProj


class LayerNormFP16(nn.LayerNorm):
    """优化的LayerNorm，处理fp16精度问题"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            return super().forward(x.float()).half()
        return super().forward(x)


class ModifiedCLIP(nn.Module):
    """
    改进的CLIP模型，支持可学习的提示词和特征分离模块
    
    Args:
        backbone: CLIP模型骨干网络
        device: 计算设备
        prompt_length: 提示词长度
        text_adapt_until: 文本适配器层数
        enable_fsm: 是否启用特征分离模块
        classnames: 类别名称列表
    """
    
    def __init__(
        self,
        backbone: str = "ViT-L/14",
        prompt_length: int = 16,
        text_adapt_until: int = 3,
        enable_fsm: bool = True,
        classnames: Optional[List[str]] = None
    ):
        super().__init__()

        self.enable_fsm = enable_fsm
        self.prompt_length = prompt_length
        self.logit_scale = torch.tensor(4.605170249938965,requires_grad=False)
        
        # 默认类别名称
        if classnames is None:
            classnames = ["real", "fake"]
        self.classnames = classnames
        self.n_cls = len(classnames)

        # 加载并冻结CLIP模型
        self._load_and_freeze_clip(backbone)
        
        # 初始化可学习的提示词
        self._initialize_learnable_prompts()
        
        # 初始化文本适配器
        self._initialize_text_adapter(text_adapt_until)
        
        # 初始化特征分离模块
        if enable_fsm:
            self._initialize_fsm()

    def _load_and_freeze_clip(self, backbone: str) -> None:
        """加载并完全冻结CLIP模型（视觉和文本编码器）"""
        self.clip, self.preprocess = load(backbone, device="cpu")
        
        # 完全冻结CLIP模型的所有参数
        for name, param in self.clip.named_parameters():
            param.requires_grad = False
            
        # 确保视觉编码器冻结
        for param in self.clip.visual.parameters():
            param.requires_grad = False
            
        # 确保文本编码器冻结  
        for param in self.clip.transformer.parameters():
            param.requires_grad = False
            
        # 冻结其他组件
        self.clip.token_embedding.requires_grad_(False)
        self.clip.positional_embedding.requires_grad_(False)
        self.clip.ln_final.requires_grad_(False)
        if hasattr(self.clip, 'text_projection') and self.clip.text_projection is not None:
            self.clip.text_projection.requires_grad_(False)
        if hasattr(self.clip, 'logit_scale'):
            self.clip.logit_scale.requires_grad_(False)

        # 注册前向钩子获取中间层输出
        self.intermediate_outputs = []
        for block in self.clip.visual.transformer.resblocks:
            block.register_forward_hook(self._hook_fn)
            
        print("CLIP模型已完全冻结（视觉和文本编码器）")

    def _hook_fn(self, module, input, output) -> None:
        """钩子函数，收集中间层输出"""
        self.intermediate_outputs.append(output)

    def _initialize_learnable_prompts(self, ctx_init: str = "") -> None:
        """初始化可学习的提示词向量"""
        dtype = self.clip.dtype
        ctx_dim = self.clip.ln_final.weight.shape[0]
        n_ctx = self.prompt_length

        if ctx_init:
            # 使用给定单词初始化上下文向量
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.clip.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 随机初始化类别特定的上下文向量
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        # 构建提示词模板
        processed_classnames = [name.replace("_", " ") for name in self.classnames]
        prompts = [f"{prompt_prefix} {name}." for name in processed_classnames]

        # 预计算固定的token向量
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.clip.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.tokenized_prompts = tokenized_prompts

    def _initialize_text_adapter(self, text_adapt_until: int) -> None:
        """初始化文本适配器"""
        if text_adapt_until > 0:
            adapters = []
            
            # 添加SimpleAdapter层
            for _ in range(text_adapt_until):
                adapters.append(SimpleAdapter(768, 768))
            
            # 添加最终的投影层
            adapters.append(SimpleProj(768, 768, relu=True))
            
            self.text_adapter = nn.ModuleList(adapters)
            
            # Xavier初始化
            for adapter in self.text_adapter:
                for param in adapter.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
        else:
            self.text_adapter = None

    def _initialize_fsm(self) -> None:
        """初始化特征分离模块 (Feature Separation Module)"""
        width = self.clip.visual.conv1.weight.shape[0]
        scale = width ** -0.5
        
        self.ln_post = LayerNormFP16(width)
        self.proj1 = nn.Parameter(scale * torch.randn(width, 768))
        self.w = nn.Parameter(torch.randn(1, 24, 768))
        self.proj2 = nn.Parameter(scale * torch.randn(768, 768))

    def _process_image_features_intermediate_outputs_with_fsm(self) -> torch.Tensor:
        dtype = self.clip.dtype
        """使用特征分离模块处理图像特征"""
        # 整理中间层输出: [24, 257, batch_size, 1024] -> [batch_size, 24, 257, 1024]
        stacked_features = torch.stack(self.intermediate_outputs, dim=0)
        features = stacked_features.permute(2, 0, 1, 3)  # LND -> NLD
        
        # 提取CLS token特征并应用层归一化
        cls_features = self.ln_post(features[:, :, 0, :])  # [batch_size, 24, 1024]
        
        # 第一次投影
        projected_features = cls_features @ self.proj1.to(dtype)  # [batch_size, 24, 768]
        
        # 应用注意力权重
        attention_weights = torch.softmax(self.w.to(dtype), dim=1)
        weighted_features = attention_weights * projected_features
        
        # 聚合特征
        aggregated_features = torch.sum(weighted_features, dim=1)  # [batch_size, 768]
        
        # 第二次投影
        final_features = aggregated_features @ self.proj2.to(dtype)  # [batch_size, 768]
        
        return final_features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            image_features: 图像特征
            text_features: 文本特征  
            logits_per_image: 图像到文本的相似度
            logits_per_text: 文本到图像的相似度
        """
        # 清空中间输出
        self.intermediate_outputs.clear()
        
        # 编码图像 - 确保无梯度计算
        with torch.no_grad():
            image_features = self.clip.encode_image(x)
        
        # 构建完整的提示词
        prompts = torch.cat([
            self.token_prefix,  # [n_cls, 1, dim]
            self.ctx,          # [n_cls, n_ctx, dim]
            self.token_suffix,  # [n_cls, *, dim]
        ], dim=1)
        
        # 编码文本 - 使用自定义的encode_text_learn确保只有adapter可训练
        text_features = self.clip.encode_text_learn(
            prompts, self.tokenized_prompts, adapter=self.text_adapter
        )
        
        # 处理图像特征
        if self.enable_fsm:
            image_features = self._process_image_features_intermediate_outputs_with_fsm()
        
        # 特征归一化
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()  # 固定的logit缩放因子
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return image_features, text_features, logits_per_image, logits_per_text, logit_scale

    def get_trainable_parameters(self) -> dict:
        """获取可训练参数的统计信息"""
        trainable_params = {}
        total_trainable = 0
        
        # 统计可学习的提示词参数
        trainable_params['learnable_prompts'] = self.ctx.numel()
        total_trainable += self.ctx.numel()
        
        # 统计文本适配器参数
        if self.text_adapter is not None:
            adapter_params = sum(p.numel() for p in self.text_adapter.parameters() if p.requires_grad)
            trainable_params['text_adapter'] = adapter_params
            total_trainable += adapter_params
        
        # 统计FSM参数
        if self.enable_fsm:
            fsm_params = (self.proj1.numel() + self.w.numel() + self.proj2.numel() + 
                         sum(p.numel() for p in self.ln_post.parameters()))
            trainable_params['fsm'] = fsm_params
            total_trainable += fsm_params
        
        trainable_params['total'] = total_trainable
        
        # 统计CLIP总参数（应该都是冻结的）
        clip_params = sum(p.numel() for p in self.clip.parameters())
        trainable_params['clip_total'] = clip_params
        trainable_params['clip_trainable'] = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        
        return trainable_params

    def get_text_features(self) -> torch.Tensor:
        """获取文本特征，用于推理时的效率优化"""
        prompts = torch.cat([
            self.token_prefix,
            self.ctx,
            self.token_suffix,
        ], dim=1)
        
        text_features = self.clip.encode_text_learn(
            prompts, self.tokenized_prompts, adapter=self.text_adapter
        )
        return text_features / text_features.norm(dim=1, keepdim=True)

    def update_classnames(self, new_classnames: List[str]) -> None:
        """动态更新类别名称"""
        if len(new_classnames) != self.n_cls:
            raise ValueError(f"新类别数量 {len(new_classnames)} 与原来的 {self.n_cls} 不匹配")
        
        self.classnames = new_classnames
        self._initialize_learnable_prompts()
        print(f"类别名称已更新为: {new_classnames}")

    def verify_clip_frozen(self) -> bool:
        """验证CLIP模型是否完全冻结"""
        frozen_params = 0
        total_params = 0
        
        for name, param in self.clip.named_parameters():
            total_params += 1
            if not param.requires_grad:
                frozen_params += 1
            else:
                print(f"警告: 参数 {name} 未冻结")
                
        is_fully_frozen = frozen_params == total_params
        print(f"CLIP冻结状态: {frozen_params}/{total_params} 参数已冻结")
        return is_fully_frozen