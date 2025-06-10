import torch.nn as nn # torch.nn 라이브러리를 nn이라는 이름으로 사용할수있도록 호출
import torchvision # torchvision 라이브러리 호출

class ResNet152(nn.Module): 
    def __init__(self, output_ch=5): # output_ch은 클래스를 의미하기 때문에 정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전 다섯개 클래스 학습을 위한 설정
        super(ResNet152, self).__init__() 
        
        self.resnet = torchvision.models.resnet152(pretrained=True) # torchvision에서 제공하는 ResNet152를 resnet 변수에 할당
        self.resnet.fc = nn.Sequential( # Sequential을 통해 사전학습된 resnet의 마지막 레이어를 본 학습 Task에 맞는 분류기 설정을 위해 output_ch = 5로 수정
                                        nn.Dropout(p=0.5, inplace=True), # Droptout 추가 
                                        nn.Linear(2048, 1024), # nn.linear를 통한 분류기 첫 번째 계층 추가
                                        nn.ReLU(inplace=True), # 활성함수 ReLU 추가
                                        nn.Linear(1024, output_ch) # nn.linear를 통한 분류기 두 번째 계층 추가
                                        )
    def forward(self, x): 
        return self.resnet(x) #resnet의 forward 결과 반환


class EfficientNet_b0(nn.Module): 
    def __init__(self, output_ch=5): # output_ch은 클래스를 의미하기 때문에 정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전 다섯개 클래스 학습을 위한 설정
        super(EfficientNet_b0, self).__init__() 
        
        self.efficientnet = torchvision.models.efficientnet_b0(pretrained=True) # torchvision에서 제공하는 efficientnet_b0를 efficientnet 변수에 할당
        self.efficientnet.classifier = nn.Sequential( # Sequential을 통해 사전학습된 efficientnet_b0의 마지막 레이어를 본 학습 Task에 맞는 분류기 설정을 위해 output_ch = 5로 수정
                                                    nn.Dropout(p=0.2, inplace=True),# Droptout 추가
                                                    nn.Linear(in_features=1280, out_features=512, bias=True), # nn.linear를 통한 분류기 첫 번째 계층 추가
                                                    nn.ReLU(inplace=True), # 활성함수 ReLU 추가
                                                    nn.Linear(in_features=512, out_features=output_ch, bias=True) # nn.linear를 통한 분류기 두 번째 계층 추가
                                                    )
    def forward(self, x):
        return self.efficientnet(x) # efficientnet의 forward 결과 반환

# class ConvNeXt_Base(nn.Module):
#     def __init__(self, output_ch=5):
#         super().__init__()
#         self.backbone = torchvision.models.convnext_base(pretrained=True)
#         # 안전하게 마지막 Linear 추출
#         last = self.backbone.classifier[-1]  # 마지막 모듈(Linear)
#         feat_dim = last.in_features
#         # 새 classifier
#         self.backbone.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(feat_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, output_ch)
#         )

#     def forward(self, x):
#         return self.backbone(x)

class ConvNeXt_Base(nn.Module):
    def __init__(self, output_ch=5):
        super().__init__()
        self.backbone = torchvision.models.convnext_base(pretrained=True)
        
        # 기존의 classifier는 AvgPool2d + Flatten + Linear로 구성되어 있음
        feat_dim = self.backbone.classifier[2].in_features
        
        # classifier 재정의 (AvgPool2d를 반드시 포함!)
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling 추가
            nn.Flatten(),                  # flatten 추가
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
        # 안전하게 마지막 Linear 입력 차원 추출
        last = self.backbone.classifier[-1]
        feat_dim = last.in_features
        # 새로운 분류기 교체
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
        # heads.head 이 마지막 Linear
        last = self.backbone.heads[-1]    # 마지막 Linear 레이어
        feat_dim = last.in_features        # 768
        self.backbone.heads = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feat_dim, output_ch)
        )

    def forward(self, x):
        return self.backbone(x)
