import torch
import torch.nn as nn
import torchvision.models as models
import time

# 创建一个稍微复杂一点的CNN模型
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 创建一个随机输入
input_data = torch.randn(1, 3, 32, 32)

# 创建CPU上的模型和CUDA上的模型
cpu_model = MyCNN()
cuda_model = MyCNN().cuda()

cpu_total_time = 0
cuda_total_time = 0

# 进行十次测试并计算平均推理时间
for i in range(10):
    # 在CPU上进行推理，并计算推理时间
    start_time = time.time()
    cpu_output = cpu_model(input_data)
    cpu_inference_time = time.time() - start_time
    cpu_total_time += cpu_inference_time

    # 在CUDA上进行推理，并计算推理时间
    start_time = time.time()
    cuda_output = cuda_model(input_data.cuda())
    torch.cuda.synchronize()
    cuda_inference_time = time.time() - start_time
    cuda_total_time += cuda_inference_time

print("CPU平均推理时间：", cpu_total_time / 10)
print("CUDA平均推理时间：", cuda_total_time / 10)
