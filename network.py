import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F



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



###############
# for Filter  #
###############
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



###############
#  for U-net  #
###############
class DoubleConv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(nin, nout, 3, padding=1, stride=1),
                                         nn.BatchNorm2d(nout),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(nout, nout, 3, padding=1, stride=1),
                                         nn.BatchNorm2d(nout),
                                         nn.ReLU(inplace=True)
                                         )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.down_conv = nn.Sequential(nn.MaxPool2d(2),
                                       DoubleConv(nin, nout))

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(nin, nout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, nin, nout):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
