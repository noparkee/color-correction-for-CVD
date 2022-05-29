import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import wandb
from collections import OrderedDict



class ResNet(torch.nn.Module):
    """ ResNet with the softmax chopped off and the batchnorm frozen """
    def __init__(self):
        super(ResNet, self).__init__()
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048
        
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()

    def forward(self, x):
        """ encode x into a feature vector of size n_outputs """
        return self.network(x)

    def train(self, mode=True):
        """ override the default train() to freeze the BN parameters """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class Identity(nn.Module):
    """ identity layer """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



class Filter(nn.Module):
    def __init__(self, device):
        super(Filter, self).__init__()

        self.device = device
        
        filter_layer = torch.empty(3, 3)
        #filter_layer2 = torch.empty(3, 3)
        torch.nn.init.normal_(filter_layer)
        #torch.nn.init.normal_(filter_layer2)
        ### learnable
        self.filter_layer = torch.nn.Parameter(filter_layer)
        #self.filter_layer2 = torch.nn.Parameter(filter_layer2)

        self.rgb2lms = torch.Tensor([[17.8824, 43.5161, 4.11935], [3.45565, 27.1554, 3.86714], [0.0299566, 0.184309, 1.46709]]).to(device)
        self.inverted_rgb2lms = torch.linalg.inv(self.rgb2lms)
        # D
        self.cvd_matrix = torch.Tensor([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]).to(device)

        self.sg = nn.Sigmoid()

    def apply_cvd(self, x):
        product1 = torch.matmul(self.inverted_rgb2lms, self.cvd_matrix)
        product2 = torch.matmul(product1, self.rgb2lms)

        B, C, H, W = x.shape

        cvd_image = torch.matmul(product2, x.reshape(B, C, -1))
        cvd_image = cvd_image.reshape(B, C, H, W)

        return cvd_image

    def apply_filter(self, x):
        B, C, H, W = x.shape

        product1 = torch.matmul(self.inverted_rgb2lms, self.filter_layer)
        product2 = torch.matmul(product1, self.rgb2lms)
        filtered_image = torch.matmul(product2, x.reshape(B, C, -1))
        filtered_image = filtered_image.reshape(B, C, H, W)

        ###

        #product2 = torch.matmul(self.inverted_rgb2lms, self.filter_layer2)
        #product1 = torch.matmul(self.filter_layer, self.rgb2lms)
        #product = torch.matmul(product2, self.sg(product1))
        #filtered_image = torch.matmul(product, x.reshape(B, C, -1))
        #filtered_image = filtered_image.reshape(B, C, H, W)

        return filtered_image


class OurModel(nn.Module):
    def __init__(self, device):
        super(OurModel, self).__init__()

        self.featurizer = ResNet()
        self.classifier = nn.Linear(self.featurizer.n_outputs, 10)
        
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

