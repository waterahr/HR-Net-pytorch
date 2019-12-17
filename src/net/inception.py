import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

    
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.b2_1x1_a = BasicConv2d(in_planes, n3x3red, 
                                    kernel_size=1)
        self.b2_3x3_b = BasicConv2d(n3x3red, n3x3, 
                                    kernel_size=3, padding=1)

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3_1x1_a = BasicConv2d(in_planes, n5x5red, 
                                    kernel_size=1)
        self.b3_3x3_b = BasicConv2d(n5x5red, n5x5, 
                                    kernel_size=3, padding=1)
        self.b3_3x3_c = BasicConv2d(n5x5, n5x5, 
                                    kernel_size=3, padding=1)

        # 3x3 pool -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_1x1 = BasicConv2d(in_planes, pool_planes, 
                                  kernel_size=1)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_3x3_b(self.b2_1x1_a(x))
        y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
        y4 = self.b4_1x1(self.b4_pool(x))
        return torch.cat([y1, y2, y3, y4], 1)
    
class GoogLeNet(nn.Module):
    def __init__(self, classes):
        super(GoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(3, 192, kernel_size=3, padding=1)

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1024, classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.sigmoid(out)
    
class HRGoogLeNet(nn.Module):
    def __init__(self, classes):
        super(HRGoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(3, 192, kernel_size=3, padding=1)

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        
        self.conv_low = BasicConv2d(512, 1024, kernel_size=3, padding=1)
        self.conv_mid = BasicConv2d(528, 1024, kernel_size=3, padding=1)
        self.conv_high = BasicConv2d(1024, 1024, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_low = nn.Linear(1024, classes[0])
        self.fc_mid = nn.Linear(1024, classes[1])
        self.fc_high = nn.Linear(1024, classes[2])
        self.fc_reason = nn.Linear(classes[0] + classes[1], classes[2])

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        
        #print("********", out.size())
        fea_low = self.conv_low(out)
        fea_low = self.avgpool(fea_low)
        fea_low = fea_low.view(fea_low.size(0), -1)
        out_low = self.fc_low(fea_low)
        
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        
        #print("********", out.size())
        fea_mid = self.conv_mid(out)
        fea_mid = self.avgpool(fea_mid)
        fea_mid = fea_mid.view(fea_mid.size(0), -1)
        out_mid = self.fc_mid(fea_mid)
        
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        
        #print("********", out.size())
        fea_high = self.conv_high(out)
        fea_high = self.avgpool(fea_high)
        fea_high = fea_high.view(fea_high.size(0), -1)
        out_high = self.fc_high(fea_high)
        
        priori = torch.cat([out_low, out_mid], 1)
        priori = F.sigmoid(priori)
        post = self.fc_reason(priori)
        post = F.sigmoid(post)
        out_high = out_high * post
        
        return F.sigmoid(torch.cat([out_low, out_mid, out_high], 1))
    
if __name__ == "__main__":
    print("OK!")