import torch
import torch.nn as nn
from torch.autograd import Variable

class FusedBlock(nn.Module):
    def __init__(self, z_dim=128):
        super(FusedBlock, self).__init__()
        self.fc = nn.Linear(128, 4*4*1024)
        self.bn1 = nn.BatchNorm1d(4*4*1024)
        
        self.conv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        h = self.relu(self.bn1(self.fc(x.view(-1, 128))))
        h = h.view(-1, 1024, 4, 4)
        h = self.relu(self.bn2(self.conv1(h)))
        return h
    
class UnconditionG(nn.Module):
    def __init__(self, num_featrues=512, img_size=128):
        super(UnconditionG, self).__init__()
        self.img_size = img_size
        
        # 8x8 --> 16x16
        self.conv1 = nn.ConvTranspose2d(num_featrues, num_featrues//2, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(num_featrues//2)
        
        # 16x16 --> 32x32
        self.conv2 = nn.ConvTranspose2d(num_featrues//2, num_featrues//4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(num_featrues//4)
         
        if img_size == 128:
            # 32x32 --> 64x64
            self.conv3 = nn.ConvTranspose2d(num_featrues//4, num_featrues//8, 4, 2, 1)
            self.bn3 = nn.BatchNorm2d(num_featrues//8)
            
            # 64x64 --> 128x128
            self.conv4 = nn.ConvTranspose2d(num_featrues//8, 3, 4, 2, 1)
        else:
            # 32x32 --> 64x64
            self.conv3 = nn.ConvTranspose2d(num_featrues//4, 3, 4, 2, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        
        if self.img_size == 128:
            h = self.relu(self.bn3(self.conv3(h)))
            out = self.tanh(self.conv4(h))
        else:
            out = self.tanh(self.conv3)
            
        return out
    
class Condition_Embedding(nn.Module):
    def __init__(self, num_attr=40):
        super(Condition_Embedding, self).__init__()
        self.num_attr = num_attr
        self.fc = nn.Linear(num_attr, num_attr*2, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def embedding(self, x):
        e = self.relu(self.fc(x))
        mu = e[:, :self.num_attr]
        logvar = e[:, self.num_attr:]
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, logvar = self.embedding(x)
        attr_code = self.reparametrize(mu, logvar)
        return attr_code, mu, logvar
    
class ConditionG(nn.Module):
    def __init__(self, num_featrues=512, img_size=128, num_attr=40):
        super(ConditionG, self).__init__()
        self.img_size = img_size
        self.condition_embedding = Condition_Embedding(num_attr=num_attr)
        
        # 8x8 --> 16x16
        self.conv1 = nn.ConvTranspose2d(num_featrues+num_attr, num_featrues//2, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(num_featrues//2)
        
        # 16x16 --> 32x32
        self.conv2 = nn.ConvTranspose2d(num_featrues//2, num_featrues//4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(num_featrues//4)
         
        if img_size == 128:
            # 32x32 --> 64x64
            self.conv3 = nn.ConvTranspose2d(num_featrues//4, num_featrues//8, 4, 2, 1)
            self.bn3 = nn.BatchNorm2d(num_featrues//8)
            
            # 64x64 --> 128x128
            self.conv4 = nn.ConvTranspose2d(num_featrues//8, 3, 4, 2, 1)
        else:
            # 32x32 --> 64x64
            self.conv3 = nn.ConvTranspose2d(num_featrues//4, 3, 4, 2, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x, y):
        attr_code, mu, logvar = self.condition_embedding(y)
        attr_code = attr_code[:,:,None,None].repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, attr_code), dim=1)
        
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        
        if self.img_size == 128:
            h = self.relu(self.bn3(self.conv3(h)))
            out = self.tanh(self.conv4(h))
        else:
            out = self.tanh(self.conv3)
            
        return out, mu, logvar
    
class Generator(nn.Module):
    def __init__(self, z_dim=128, num_featrues=512, img_size=128, num_attr=40):
        super(Generator, self).__init__()
        self.fused_block = FusedBlock(z_dim=z_dim)
        self.uncondition_generator = UnconditionG(num_featrues=num_featrues, img_size=img_size)
        self.condition_generator = ConditionG(num_featrues=num_featrues, img_size=img_size, num_attr=num_attr)
        
    def forward(self, x, y):
        h = self.fused_block(x)
        hu = self.uncondition_generator(h)
        hc, mu, logvar = self.condition_generator(h, y)
        return hu, hc, mu, logvar
    
#### Discriminator
class UnconditionD(nn.Module):
    def __init__(self, num_featrues=64, img_size=128):
        super(UnconditionD, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(3, num_featrues, 4, 2, 1, bias=False)
        
        self.conv2 = nn.Conv2d(num_featrues, num_featrues*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_featrues*2)
        
        self.conv3 = nn.Conv2d(num_featrues*2, num_featrues*4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_featrues*4)
        
        self.conv4 = nn.Conv2d(num_featrues*4, num_featrues*8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_featrues*8)
        
        if img_size == 128:
            self.conv5 = nn.Conv2d(num_featrues*8, num_featrues*16, 4, 2, 1, bias=False)
            self.bn5 = nn.BatchNorm2d(num_featrues*16)
            
            self.conv6 = nn.Conv2d(num_featrues*16, 1, 4, 1, 0, bias=False)
            
        else:
            self.conv5 = nn.Conv2d(num_featrues*16, 1, 4, 1, 0, bias=False)
            
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        
    def forward(self, x):
        h = self.lrelu(self.conv1(x))
        h = self.lrelu(self.bn2(self.conv2(h)))
        h = self.lrelu(self.bn3(self.conv3(h)))
        h = self.lrelu(self.bn4(self.conv4(h)))
        
        if self.img_size == 128:
            h = self.lrelu(self.bn5(self.conv5(h)))
            out = self.conv6(h)
        else:
            out = self.conv5(h)
        
        return out.view(-1)
    
class ConditionD(nn.Module):
    def __init__(self, num_featrues=64, img_size=128, num_attr=40):
        super(ConditionD, self).__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(3, num_featrues, 4, 2, 1, bias=False)
        
        self.conv2 = nn.Conv2d(num_featrues, num_featrues*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_featrues*2)
        
        self.conv3 = nn.Conv2d(num_featrues*2, num_featrues*4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_featrues*4)

        if img_size == 128:
            attr_dim = 0
        else:
            attr_dim = 40
        self.conv4 = nn.Conv2d(num_featrues*4+attr_dim, num_featrues*8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_featrues*8)
        
        if img_size == 128:
            attr_dim = num_attr
            self.conv5 = nn.Conv2d(num_featrues*8+attr_dim, num_featrues*16, 4, 2, 1, bias=False)
            self.bn5 = nn.BatchNorm2d(num_featrues*16)
            
            self.conv6 = nn.Conv2d(num_featrues*16, 1, 4, 1, 0, bias=False)
            
        else:
            self.conv5 = nn.Conv2d(num_featrues*16, 1, 4, 1, 0, bias=False)
            
        self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        
    def forward(self, x, y):
        y = y[:,:,None,None].repeat(1,1,8,8)
        h = self.lrelu(self.conv1(x))
        h = self.lrelu(self.bn2(self.conv2(h)))
        h = self.lrelu(self.bn3(self.conv3(h)))
        if self.img_size is not 128:
            h = torch.cat((h, y), dim=1)
        h = self.lrelu(self.bn4(self.conv4(h)))
        
        if self.img_size == 128:
            h = torch.cat((h, y), dim=1)
            h = self.lrelu(self.bn5(self.conv5(h)))
            out = self.conv6(h)
        else:
            out = self.conv5(h)
        
        return out.view(-1)
        
            