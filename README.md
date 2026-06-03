# PromptFake

Generalizable Deepfake Detection via Orthogonal Prompts and Layer-Wise Feature Decoupling in CLIP

## Overview

PromptFake is a novel deepfake detection method that leverages the power of CLIP (Contrastive Language-Image Pre-training) models enhanced with orthogonal prompts and layer-wise feature decoupling techniques. This approach aims to improve the generalization ability of deepfake detection models across various generative methods, including GANs, diffusion models, and other AI-generated content.

## Features

- **Orthogonal Prompts**: Uses learnable prompts to guide the CLIP model in distinguishing between real and fake images
- **Layer-wise Feature Decoupling**: Separates features at different layers to capture multi-scale discriminative patterns
- **Generalizability**: Designed to detect various types of synthetic content from different generation methods
- **CLIP Integration**: Leverages pre-trained CLIP models for robust visual-language representations
- **Comprehensive Evaluation**: Tested on multiple datasets including StyleGAN, ProGAN, BigGAN, diffusion models, and more

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/fjnu-cxy/PromptFake.git
   cd PromptFake
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Methodology

Our approach introduces two key innovations:

1. **Orthogonal Prompts**: Learnable textual prompts that guide the CLIP model to focus on discrepancies between real and synthetic content
2. **Layer-wise Feature Decoupling**: A mechanism to separate and analyze features at different layers of the CLIP model to capture multi-level artifacts

The model is implemented as a modified version of CLIP with:
- Learnable prompt tokens
- Adapter modules for fine-tuning
- Feature separation mechanisms

## Usage

### Training

To train the model:

```bash
python train.py --data_dir /path/to/data --backbone ViT-L/14 --prompt_length 16 --text_adapt_until 3
```

Available options:
- `--data_dir`: Path to training data directory
- `--backbone`: CLIP model architecture (e.g., ViT-L/14, ViT-B/32)
- `--prompt_length`: Length of learnable prompts
- `--text_adapt_until`: Number of text transformer layers to adapt
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate

### Testing

To evaluate the trained model:

```bash
python test.py --model_path /path/to/checkpoint --data_dir /path/to/test_data
```

## Datasets

The model is evaluated on diverse synthetic image sources:

- GAN-based: ProGAN, StyleGAN, StyleGAN2, BigGAN
- Diffusion models: GLIDE, Latent Diffusion Models (LDM), DALL-E 2
- Other generative models: CycleGAN, StarGAN, GauGAN
- Deepfakes: FaceSwap, DeepFaceLab

JSON configuration files for each dataset are located in the `data/` directory.

## Checkpoints

Trained model checkpoints are saved in the `checkpoints/` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
