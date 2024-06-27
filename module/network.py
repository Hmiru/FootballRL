
from utils import *

class SuccessPredictionNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SuccessPredictionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * input_shape[0] * input_shape[1], 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, output_shape[0] * output_shape[1])
        self.output_shape = output_shape

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)  # view에서 reshape로 변경
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid를 사용하여 확률값으로 변환
        return x.reshape(-1, *self.output_shape)  # view에서 reshape로 변경

if __name__ == "__main__":
    input_shape = (80, 120, 11)
    output_shape = (80, 120)
    net = SuccessPredictionNetwork(input_shape, output_shape)
    print(net)
