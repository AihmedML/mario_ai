import torch
import torch.nn as nn

NUM_AUX = 3  # x_pos, y_pos, time


class MarioNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(6400 + NUM_AUX, 512), nn.ReLU())
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x, aux=None):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if aux is not None:
            x = torch.cat([x, aux], dim=1)
        else:
            x = torch.cat([x, torch.zeros(x.size(0), NUM_AUX, device=x.device)], dim=1)
        x = self.fc(x)
        return self.policy(x), self.value(x)
