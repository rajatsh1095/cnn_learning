import torch
import torch.nn as nn
import torch.nn.functional as F

conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

x = torch.randn(1, 3, 224, 224)

out = conv(x)
out = F.relu(out)
out = pool(out)

out = conv2(out)
out = F.relu(out)
out = pool(out)


out = out.view(out.size(0), -1)
print(out.shape)

fc1 = nn.Linear(50176, 128)
fc2 = nn.Linear(128, 2)

out = F.relu(fc1(out))
out = fc2(out)

print(out.shape)