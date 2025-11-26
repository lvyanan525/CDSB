import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, feature_map, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feature_map * gamma + beta

class BrightNormNet(nn.Module):
    def __init__(self, brightness_param_dim, input_channels=3, channels=64, num_blocks=3):
        super().__init__()
        
        self.conv_in = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
        self.film_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.film_blocks.append(nn.ModuleDict(
                {
                    "conv": nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    "filmlayer": FiLMLayer(channels),
                    "act": nn.SiLU(),
                }
            ))
        self.conv_out = nn.Conv2d(channels, input_channels, kernel_size=1)

        self.cond_proj = nn.Sequential(
            nn.Linear(brightness_param_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_blocks * channels * 2),
        )

        self.channels = channels
        self.num_blocks = num_blocks
        self.input_channels = input_channels

    def forward(self, image, brightness_params):
        data_channels = image.shape[1]
        if data_channels == 1 and self.input_channels == 3:
            image = image.repeat(1, 3, 1, 1)
        elif data_channels == 3 and self.input_channels == 1:
            image = image.mean(dim=1, keepdim=True)
        elif data_channels != self.input_channels:
            raise ValueError(f"Input channels {data_channels} does not match input channels {self.input_channels}")
        
        x = self.conv_in(image)
        brightness_params = self.cond_proj(brightness_params)
        p = brightness_params.view(-1, self.num_blocks, self.channels * 2)
        brightness_params = p.permute(1, 0, 2)
        

        for i, block in enumerate(self.film_blocks):
            gamma, beta = torch.chunk(brightness_params[i], 2, dim=1)
            x = block["conv"](x)
            x = block["filmlayer"](x, gamma, beta)
            x = block["act"](x)

        x = self.conv_out(x)
        out = image + x
        if data_channels == 1 and self.input_channels == 3:
            out = out.mean(dim=1, keepdim=True)
        elif data_channels == 3 and self.input_channels == 1:
            out = out.repeat(1, 3, 1, 1)
        return out

