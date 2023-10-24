import torch
import torch.nn as nn
import torchvision
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, EfficientNet_V2_L_Weights, \
    EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights, ResNet152_Weights, ResNet101_Weights, ResNet50_Weights, \
    Wide_ResNet101_2_Weights, Wide_ResNet50_2_Weights, ViT_B_16_Weights, Swin_V2_B_Weights, Swin_V2_S_Weights, \
    Swin_V2_T_Weights


class PretrainedModelsFeature(nn.Module):
    def __init__(self, pretrained_model=""):
        super(PretrainedModelsFeature, self).__init__()
        self.last_fea = None
        if pretrained_model == "mobilenet_v3_large":
            self.backbone = torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            self.backbone_name = "mobilenet_v3_large"
            print("using mobilenet_v3_large with MobileNet_V3_Large_Weights.IMAGENET1K_V2")
        elif pretrained_model == "mobilenet_v3_small":
            self.backbone = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.backbone_name = "mobilenet_v3_small"
            print("using mobilenet_v3_small with MobileNet_V3_Small_Weights.IMAGENET1K_V1")
        elif pretrained_model == "efficientnet_v2_l":
            self.backbone = torchvision.models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            self.backbone_name = "efficientnet_v2_l"
            print("using efficientnet_v2_l with EfficientNet_V2_L_Weights.IMAGENET1K_V1")
        elif pretrained_model == "efficientnet_v2_m":
            self.backbone = torchvision.models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            self.backbone_name = "efficientnet_v2_m"
            print("using efficientnet_v2_m with EfficientNet_V2_M_Weights.IMAGENET1K_V1")
        elif pretrained_model == "efficientnet_v2_s":
            self.backbone = torchvision.models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.backbone_name = "efficientnet_v2_s"
            print("using efficientnet_v2_s with EfficientNet_V2_S_Weights.IMAGENET1K_V1")
        elif pretrained_model == "resnet152":
            self.backbone = torchvision.models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            self.backbone_name = "resnet152"
            print("using resnet152 with ResNet152_Weights.IMAGENET1K_V2")
        elif pretrained_model == "resnet101":
            self.backbone = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            self.backbone_name = "resnet101"
            print("using resnet101 with ResNet101_Weights.IMAGENET1K_V2")
        elif pretrained_model == "resnet50":
            self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.backbone_name = "resnet50"
            print("using resnet50 with ResNet50_Weights.IMAGENET1K_V2")
        elif pretrained_model == "wide_resnet101_2":
            self.backbone = torchvision.models.wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V2)
            self.backbone_name = "wide_resnet101_2"
            print("using wide_resnet101_2 with Wide_ResNet101_2_Weights.IMAGENET1K_V2")
        elif pretrained_model == "wide_resnet50_2":
            self.backbone = torchvision.models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
            self.backbone_name = "wide_resnet50_2"
            print("using wide_resnet50_2 with Wide_ResNet50_2_Weights.IMAGENET1K_V2")
        elif pretrained_model == "vit_b_16":
            self.backbone = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            self.backbone_name = "vit_b_16"
            print("using vit_b_16 with ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1")
        elif pretrained_model == "swin_v2_b":
            self.backbone = torchvision.models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            self.backbone_name = "swin_v2_b"
            print("using swin_v2_b with Swin_V2_B_Weights.IMAGENET1K_V1")
        elif pretrained_model == "swin_v2_s":
            self.backbone = torchvision.models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
            self.backbone_name = "swin_v2_s"
            print("using swin_v2_s with Swin_V2_S_Weights.IMAGENET1K_V1")
        elif pretrained_model == "swin_v2_t":
            self.backbone = torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
            self.backbone_name = "swin_v2_t"
            print("using swin_v2_t with Swin_V2_T_Weights.IMAGENET1K_V1")
        else:
            raise Exception("Pretrained models not available for now, please implement yourself")

        if pretrained_model in ["mobilenet_v3_large", "mobilenet_v3_small"]:
            self.backbone.classifier = nn.Sequential()
        elif pretrained_model in ["efficientnet_v2_l", "efficientnet_v2_m", "efficientnet_v2_s"]:
            self.backbone.classifier = nn.Sequential()
        elif pretrained_model in ['resnet152', "resnet101", "resnet50"]:
            self.backbone.fc = nn.Sequential()
        elif pretrained_model in ["wide_resnet101_2", "wide_resnet50_2"]:
            self.backbone.fc = nn.Sequential()
        elif pretrained_model in ["vit_b_16"]:
            self.backbone.heads = nn.Sequential()
        elif pretrained_model in ["swin_v2_b", "swin_v2_s"]:
            self.backbone.head = nn.Sequential()
        else:
            raise Exception("the pretrained model is not impelemted yet!")

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = PretrainedModelsFeature(pretrained_model="vit_b_16")
    print(model)

