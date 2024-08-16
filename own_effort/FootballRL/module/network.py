import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class SoccerMap(nn.Module):
    def __init__(self, input_channels=11):
        super(SoccerMap, self).__init__()

        # 1x scale layers
        self.conv1x_1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.conv1x_2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1/2x scale layers
        self.conv1_2x_1 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv1_2x_2 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1/4x scale layers
        self.conv1_4x_1 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv1_4x_2 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)

        # 1x1 convolution for reducing channels
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)

        # Upsample layers with additional convolutions
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.upsample_conv1_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_conv2_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.upsample_conv2_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self.pred_conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)
        self.pred_conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.pred_conv3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)

        self.fusion_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x):
        # 1x scale
        x1 = F.relu(self.conv1x_1(x)) # 11*80*120 -> 64*80*120
        x1 = F.relu(self.conv1x_2(x1)) # 64*80*120 -> 128*80*120
        p1 = self.pool1(x1) # 128*80*120 -> 128*40*60

        # 1/2x scale
        x2 = F.relu(self.conv1_2x_1(p1)) # 64*40*60 -> 128*40*60
        x2 = F.relu(self.conv1_2x_2(x2)) # 128*40*60 -> 128*40*60
        p2 = self.pool2(x2) # 128*40*60 -> 128*20*30

        # 1/4x scale
        x3 = F.relu(self.conv1_4x_1(p2)) # 128*20*30 -> 256*20*30
        x3 = F.relu(self.conv1_4x_2(x3)) # 256*20*30 -> 128*20*30

        # Prediction at 1/4x scale
        pred1 = F.relu(self.pred_conv1(x1)) # 128*80*120 -> 32*80*120
        pred1 = self.pred_conv2(pred1)  # 32*80*120 -> 1*80*120

        pred2 = F.relu(self.pred_conv1(x2)) # 128*40*60 -> 32*40*60
        pred2 = self.pred_conv2(pred2)  # 32*40*60 -> 1*40*60

        pred3 = F.relu(self.pred_conv1(x3)) # 128*20*30 -> 32*20*30
        pred3 = self.pred_conv2(pred3) # 32*20*30 -> 1*20*30

        # Upsample and concatenate
        u1 = self.upsample1(pred3) # 1*20*30 -> 1*40*60
        u1 = F.relu(self.upsample_conv1_1(u1))  # 1*40*60 -> 32*40*60
        u1 = self.upsample_conv1_2(u1)  # 32*40*60 -> 1*40*60

        cat1 = torch.cat([u1, pred2], dim=1) # 1*40*60 + 1*40*60 -> 2*40*60
        fus1=self.fusion_conv(cat1) # 2*40*60 -> 1*40*60

        u2 = self.upsample2(fus1) # 1*40*60 -> 1*80*120
        u2=F.relu(self.upsample_conv2_1(u2)) # 1*80*120 -> 32*80*120
        u2=self.upsample_conv2_2(u2) # 32*80*120 -> 1*80*120

        cat2 = torch.cat([u2, pred1], dim=1) # 1*80*120 + 1*80*120 -> 2*80*120
        fus2=self.fusion_conv(cat2)# 2*80*120 -> 1*80*120

        pred4 = F.relu(self.pred_conv3(fus2)) # 1*80*120 -> 32*80*120
        pred4 = self.pred_conv2(pred4) # 32*80*120 -> 1*80*120

        #sigmoid
        pass_map = torch.sigmoid(pred4)

        return pass_map


if __name__ == "__main__":
    input_size = 11
    model = SoccerMap(input_size).cuda()
    summary(model, (input_size, 80, 120))
