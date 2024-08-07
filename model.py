import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from haralick import hara_loss

class Self_Attn(nn.Module):
    '''Self attention Block'''
    def __init__(self, in_ch, out_ch):
        super(Self_Attn,self).__init__()
        self.C = in_ch
        #self.N = N
        #self.H = H
        #self.W = W
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        '''
        self.query_conv = nn.Sequential(
            nn.Conv2d(self.in_ch,self.out_ch,kernel_size=1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(self.in_ch,self.out_ch,kernel_size=1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(self.in_ch,self.out_ch,kernel_size=1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )
        '''
        self.query_conv = nn.Conv2d(self.in_ch,self.out_ch,kernel_size=1)  # 查询向量
        self.key_conv = nn.Conv2d(self.in_ch,self.out_ch,kernel_size=1)    # 键值对注意力机制
        self.value_conv = nn.Conv2d(self.in_ch,self.out_ch,kernel_size=1)

    def forward(self,x):
        '''
            inputs:
                x : input feature maps(N * C * H * W)
            returns:
                out : self attention value + input feature
        '''
        '''
        self.H = x.shape[-1]
        self.W = x.shape[-1]

        query = self.query_conv(x)
        #print('q = ',query.size())
        query = query.view(-1,self.C,1,self.H*self.W)

        key = self.key_conv(x)
        #print('k = ',key.size())
        key = key.view(-1,self.C,1,self.H*self.W)

        value = self.value_conv(x)
        #print('v = ',value.size())
        value = value.view(-1,self.C,1,self.H*self.W)
        '''
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        sigma = torch.matmul(query.permute(0,1,3,2),key)
        attention = F.softmax(sigma,dim=3)#按行softmax
        #out = torch.matmul(value,attention).view(-1,self.C,self.H,self.W)
        out = torch.matmul(value,attention)
        #print('out = ',out.size())
        #print('x = ',x.size())
        out = self.gamma*out + x
        
        return out, self.gamma
'''
class MHSA(nn.Moudle):
    def __init__(self, in_ch, out_ch, N):
        super(MHSA,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.N = N
        self.w_conv = nn.Conv(self.in_ch,self.out_ch,kernel_size=1)
        self.attenion1 = Self_Attn(self.in_ch,self.out_ch,N)
        self.attenion2 = Self_Attn(self.in_ch,self.out_ch,N)
        self.attenion3 = Self_Attn(self.in_ch,self.out_ch,N)
        self.attenion4 = Self_Attn(self.in_ch,self.out_ch,N)

    def dorward(x):
        z1 = self.attention1(x)
        z2 = self.attention2(x)
        z3 = self.attention3(x)
        z4 = self.attention4(x)
        z = torch.concat(z1,z2,z3,z4)
        out = self.w_conv(z)
'''        


