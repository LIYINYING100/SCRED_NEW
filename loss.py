import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
from torchvision.models import vgg19
from haralick import hara_loss

# from TextureEncoder import TE
from measure import updata_te

def feature_extractor(image, model):
    model.eval()
    pred, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = model(image)
    return x1, x6

class p_loss(nn.Module):  # pretrained loss
    def __init__(self):
        super(p_loss, self).__init__()
        vgg19_model = vgg19(pretrained=True)

        '''
        for param in vgg19_model.parameters():
            param.requires_grad = False
            vgg19_model_new = list(vgg19_model.features.children())[:18]
            vgg19_model_new[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.feature_extractor = nn.Sequential(*vgg19_model_new)
        '''
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])  # feature获取特征层的特征 children,只遍历模型的子层，不遍历子层的子层
        # print(list(vgg19_model.features.children())[:35])
        self.p_criterion = nn.L1Loss()  # 计算output和target之差的绝对值

    def forward_hook(mudule, inp, outp):
        feature_map = {}
        feature_map['features18'] = outp

    def forward(self, x, y):
        low = x.repeat(1,3,1,1)
        full = y.repeat(1,3,1,1)
        low_feature = self.feature_extractor(low)
        full_feature = self.feature_extractor(full)
        loss = self.p_criterion(low_feature, full_feature)
        return loss


class h_loss(nn.Module):  # haralick特征
    def __init__(self):
        super(h_loss,self).__init__()

    def forward(self, x, y):
        x = x.view(160, 64, 64)
        y = y.view(160, 64, 64)
        loss_new = torch.cuda.FloatTensor([0])
        for i in range(160):
            l = hara_loss(x[i],y[i])
            #l = l/160
            #print(l)
            loss_new += l
        loss_new = loss_new/160
        print(loss_new)
        return loss_new

class t_loss(nn.Module):
    def __init__(self, module):
        super(t_loss,self).__init__()
        self.module = module
        self.p_criterion = nn.MSELoss()
        self.lamda_1 = nn.Parameter(torch.tensor([0.01]))
        self.lamda_2 = nn.Parameter(torch.tensor([0.01]))
        #te_path = "/home/wjy/zxm/SAED_CNN/TE/1e-4_b16-psnr20/TE_13000iter.ckpt"  
        #Te = TE()
        #self.te = updata_te(Te, te_path)
        self.feature_extractor = feature_extractor
    def forward(self,x,y):
        x_b, x_t = self.feature_extractor(x, self.module)
        y_b, y_t = self.feature_extractor(y, self.module)
        loss_b = self.p_criterion(x_b, y_b)
        loss_t = self.p_criterion(x_t, y_t)

        loss = 0.7*self.p_criterion(x_b, y_b) + 0.3*self.p_criterion(x_t, y_t)

        #x_b = self.feature_extractor(x, self.module)
        #y_b = self.feature_extractor(y, self.module)
        #loss = self.p_criterion(x_b, y_b)

        return loss#,self.lamda_1,self.lamda_2

'''
    AMLN_nonlocal
'''

class CrossEntropyLossForSoftTarget(nn.Module):
    def __init__(self, T=20):
        super(CrossEntropyLossForSoftTarget, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred.div(self.T)
        y_gt_soft = y_gt.div(self.T)
        return -(self.softmax(y_gt_soft)*self.logsoftmax(y_pred_soft)).mean().mul(self.T*self.T)



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 3.5).unsqueeze(1)
    _2D_window =_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window.cuda()

class STD(torch.nn.Module):
    def __init__(self, window_size = 5):
        super(STD, self).__init__()
        self.window_size = window_size
        self.channel=1
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.window=create_window(self.window_size, self.channel)
        self.window.to(torch.device('cuda'))
    def forward(self, img):
        mu = F.conv2d(img, self.window, padding = self.window_size//2, groups = self.channel)
        mu_sq=mu.pow(2)
        sigma_sq = F.conv2d(img*img, self.window, padding = self.window_size//2, groups = self.channel) - mu_sq
        B,C,W,H=sigma_sq.shape
        sigma_sq=torch.flatten(sigma_sq, start_dim=1)
        noise_map = self.softmax(sigma_sq)
        noise_map=torch.reshape(noise_map,[B,C,W,H])
        return noise_map

class SNDisLoss(torch.nn.Module):
    """
    D Loss
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return -torch.mean(pos) + torch.mean(neg)

class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)

class NCMSE(torch.nn.Module):
    def __init__(self):
        super(NCMSE, self).__init__()
        self.std=STD()
    def forward(self, out_image, gt_image, org_image):
        loss = torch.mean(torch.mul(self.std(org_image - gt_image), torch.pow(out_image - gt_image, 2)))
        return loss


