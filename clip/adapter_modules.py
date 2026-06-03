import torch.nn as nn
import torch

class SimpleAdapter(nn.Module):
    def __init__(self, c_in, c_out=768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_out, bias=False), 
            nn.LeakyReLU()
        )

    def forward(self, x):
        if x.dtype == torch.float16:
            return self.fc(x.float()).half()
        x = self.fc(x)
        return x


class SimpleProj(nn.Module):
    def __init__(self, c_in, c_out=768, relu=True):
        super(SimpleProj, self).__init__()
        if relu:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_out, bias=False), 
                nn.LeakyReLU()
            )
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x):
        if x.dtype == torch.float16:
            return self.fc(x.float()).half()
        x = self.fc(x)
        return x