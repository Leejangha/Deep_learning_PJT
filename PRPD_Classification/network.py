import torch.nn as nn
import torchvision

class ResNet152(nn.Module): 
    def __init__(self, output_ch=5): # 정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전 다섯개 클래스
        super(ResNet152, self).__init__()
        
        self.resnet = torchvision.models.resnet152(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_ch)
        )

    def forward(self, x):
        return self.resnet(x)


class EfficientNet_b0(nn.Module): 
    def __init__(self, output_ch=5):
        super(EfficientNet_b0, self).__init__() 
        
        self.efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
                                                    nn.Dropout(p=0.2, inplace=True),
                                                    nn.Linear(in_features=1280, out_features=512, bias=True),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(in_features=512, out_features=output_ch, bias=True)
                                                    )

    def forward(self, x):
        return self.efficientnet(x)


class ConvNeXt_Base(nn.Module):
    def __init__(self, output_ch=5):
        super().__init__()
        self.backbone = torchvision.models.convnext_base(pretrained=True)
        
        # 기존의 classifier는 AvgPool2d + Flatten + Linear로 구성되어 있음
        feat_dim = self.backbone.classifier[2].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling 추가
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_ch)
        )

    def forward(self, x):
        return self.backbone(x)

    

class MobileNet_V2(nn.Module):
    def __init__(self, output_ch=5):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v2(pretrained=True)

        last = self.backbone.classifier[-1]
        feat_dim = last.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, output_ch)
        )

    def forward(self, x):
        return self.backbone(x)


class ViT_B_16(nn.Module):
    def __init__(self, output_ch=5):
        super().__init__()
        self.backbone = torchvision.models.vit_b_16(pretrained=True)

        last = self.backbone.heads[-1]
        feat_dim = last.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feat_dim, output_ch)
        )

    def forward(self, x):
        return self.backbone(x)
