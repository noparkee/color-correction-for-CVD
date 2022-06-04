import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import wandb
from collections import OrderedDict

from network import ResNet, Filter
from network import DoubleConv, Down, Up, OutConv



###############
# for Filter  #
###############
class OurModel(nn.Module):
    def __init__(self, device, num_class):
        super(OurModel, self).__init__()

        self.featurizer = ResNet()
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_class)
        
        self.filtering = Filter(device)

    def forward(self, x, y, i):
        # cvd - log용
        cvd_image = self.filtering.apply_cvd(x)

        # 어떤 필터 적용
        filtered_image = self.filtering.apply_filter(x)
        # cvd 필터 적용
        cvd_filtered_image = self.filtering.apply_cvd(filtered_image)

        ### 두 개가 큰 차이 없는거 같옹
        naturalness_loss = F.mse_loss(cvd_filtered_image, x)
        #naturalness_loss = ((cvd_filtered_image - x) ** 2).mean()

        image_features = self.featurizer(cvd_filtered_image)
        cls_outputs = self.classifier(image_features)
        cls_loss = F.cross_entropy(cls_outputs, y)

        loss = 10 * cls_loss + naturalness_loss

        ### log
        if i == 0 or i % 100 == 99:
            wandb.log({"original": [wandb.Image(x)], "cvd": [wandb.Image(cvd_image)], \
                        "filtered": [wandb.Image(filtered_image)], "cvd_filtered": [wandb.Image(cvd_filtered_image)]})

        return OrderedDict({'loss': loss, 'cls_loss': cls_loss, 'naturalness': naturalness_loss})
    
    def evaluate(self, x, y):
        # cvd - log용
        cvd_image = self.filtering.apply_cvd(x)

        # 어떤 필터 적용
        filtered_image = self.filtering.apply_filter(x)
        # cvd 필터 적용
        cvd_filtered_image = self.filtering.apply_cvd(filtered_image)

        ### 두 개가 큰 차이 없는거 같옹
        naturalness_loss = F.mse_loss(cvd_filtered_image, x)
        #naturalness_loss = ((cvd_filtered_image - x) ** 2).mean()

        image_features = self.featurizer(cvd_filtered_image)
        cls_outputs = self.classifier(image_features)
        cls_loss = F.cross_entropy(cls_outputs, y)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))
    
        wandb.log({"test_original": [wandb.Image(x)], "test_cvd": [wandb.Image(cvd_image)], \
                    "test_filtered": [wandb.Image(filtered_image)], "test_cvd_filtered": [wandb.Image(cvd_filtered_image)]})

        return correct, total



###############
#  for U-net  #
###############
class UNet(nn.Module):
    def __init__(self, device, num_class, nin=3, nout=3):
        super().__init__()
        
        self.featurizer = ResNet()
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_class)
        
        self.filtering = Filter(device)

        self.in_conv = DoubleConv(nin, 64)      # (default=1, 흑백 이미지가 들어가니까 nin=1) - 나도 흑백으로 변환하고 해야할까? 아니면 그냥 ㄱ?
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.out_conv = OutConv(64, nout)       # (default=2) rgb 채널 하고 싶어서 nout=3 으로 할래

    def forward(self, x, y, i):     # x.shape = [B, 3, 448, 448]
        cvd_image = self.filtering.apply_cvd(x)

        ### Encoder
        encode_x1 = self.in_conv(x)        # [B, 64, 448, 448]
        encode_x2 = self.down1(encode_x1)         # [4, 128, 224, 224]
        encode_x3 = self.down2(encode_x2)         # [4, 256, 112, 112]
        encode_x4 = self.down3(encode_x3)         # [4, 512, 56, 56]
        encode_x5 = self.down4(encode_x4)         # [4, 512, 28, 28]

        ### Decoder
        decode_x = self.up1(encode_x5, encode_x4)        # [4, 256, 56, 56]
        decode_x = self.up2(decode_x, encode_x3)         # [4, 128, 112, 112]
        decode_x = self.up3(decode_x, encode_x2)         # [4, 64, 224, 224]
        decode_x = self.up4(decode_x, encode_x1)         # [4, 64, 448, 448]
        decode_x = self.out_conv(decode_x)        # [4, 3, 448, 448]



        cvd_filtered_image = self.filtering.apply_cvd(decode_x)
        #cvd_encode_x5 = self.filtering.apply_cvd(encode_x5)

        naturalness_loss = F.mse_loss(cvd_filtered_image, x)                # 원본 이미지와 재생성+CVD 이미지가 유사하도록
        #embedding_naturalness_loss = F.mse_loss(cvd_encode_x5, encode_x5)   # 임베딩 레벨에서도 비슷하도록..?

        image_features = self.featurizer(cvd_filtered_image)
        cls_outputs = self.classifier(image_features)
        cls_loss = F.cross_entropy(cls_outputs, y)

        loss = cls_loss + 10 * naturalness_loss
        #loss = cls_loss + naturalness_loss + embedding_naturalness_loss

        if i == 0 or i % 100 == 99:
            wandb.log({"original": [wandb.Image(x)], "cvd": [wandb.Image(cvd_image)], \
                            "filtered": [wandb.Image(decode_x)], "cvd_filtered": [wandb.Image(cvd_filtered_image)]})

        return OrderedDict({'loss': loss, 'cls_loss': cls_loss, 'naturalness': naturalness_loss})
        #return OrderedDict({'loss': loss, 'cls_loss': cls_loss, 'naturalness': naturalness_loss, "embedding_naturalness_loss": embedding_naturalness_loss})


    def evaluate(self, x, y):
        cvd_image = self.filtering.apply_cvd(x)

        ### Encoder
        encode_x1 = self.in_conv(x)        # [B, 64, 448, 448]
        encode_x2 = self.down1(encode_x1)         # [4, 128, 224, 224]
        encode_x3 = self.down2(encode_x2)         # [4, 256, 112, 112]
        encode_x4 = self.down3(encode_x3)         # [4, 512, 56, 56]
        encode_x5 = self.down4(encode_x4)         # [4, 512, 28, 28]

        ### Decoder
        decode_x = self.up1(encode_x5, encode_x4)        # [4, 256, 56, 56]
        decode_x = self.up2(decode_x, encode_x3)         # [4, 128, 112, 112]
        decode_x = self.up3(decode_x, encode_x2)         # [4, 64, 224, 224]
        decode_x = self.up4(decode_x, encode_x1)         # [4, 64, 448, 448]
        decode_x = self.out_conv(decode_x)        # [4, 3, 448, 448]

        cvd_filtered_image = self.filtering.apply_cvd(decode_x)

        naturalness_loss = F.mse_loss(cvd_filtered_image, x)

        image_features = self.featurizer(cvd_filtered_image)
        cls_outputs = self.classifier(image_features)
        cls_loss = F.cross_entropy(cls_outputs, y)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))

        wandb.log({"original": [wandb.Image(x)], "cvd": [wandb.Image(cvd_image)], \
                    "filtered": [wandb.Image(decode_x)], "cvd_filtered": [wandb.Image(cvd_filtered_image)]})

        return correct, total
