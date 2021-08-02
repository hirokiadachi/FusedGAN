import os
import yaml
import shutil
import argparse
import numpy as np
import multiprocessing
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from dataloader import CelebA_loader
from model import Generator, UnconditionD, ConditionD

p = argparse.ArgumentParser()
p.add_argument('--cfile', '-c', type=str, default='config')
p.add_argument('--checkpoints', type=str, default='checkpoints')
p.add_argument('--gpu', '-g', type=int, nargs='+', default=[0, 1],
               help='# of GPU. (1 GPU: single GPU)')
args = p.parse_args()

##################################
# Loading training configure
##################################
with open(args.cfile) as yml_file:
    config = yaml.safe_load(yml_file.read())['training']

batch_size = config['batch_size']
start_epoch = config['start_epoch']
end_epoch = config['end_epoch']
lr = config['lr']
beta = config['beta']
weight_decay = config['weight_decay']
tb = config['tb']
img_size = config['img_size']
attr_idx = config['attr_idx']

os.makedirs(args.checkpoints, exist_ok=True)
tb_path = os.path.join(args.checkpoints, tb)
if os.path.exists(tb_path):    shutil.rmtree(tb_path)
tb = SummaryWriter(log_dir=tb_path)
device = torch.device('cuda:%d' % args.gpu[0])

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def main():
    iters = 0
    train_data = CelebA_loader(attribute_index=attr_idx)
    train_sets = DataLoader(dataset=train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=multiprocessing.cpu_count())
    
    G = nn.DataParallel(Generator(z_dim=128, num_featrues=512, img_size=img_size, num_attr=len(attr_idx)).to(device), device_ids=args.gpu)
    Du = nn.DataParallel(UnconditionD(num_featrues=64, img_size=img_size).to(device), device_ids=args.gpu)
    Dc = nn.DataParallel(ConditionD(num_featrues=64, img_size=img_size, num_attr=len(attr_idx)).to(device), device_ids=args.gpu)
    G_optim = optim.Adam(G.parameters(), lr=lr, betas=(beta[0], beta[1]))
    Du_optim = optim.Adam(Du.parameters(), lr=lr, betas=(beta[0], beta[1]))
    Dc_optim = optim.Adam(Dc.parameters(), lr=lr, betas=(beta[0], beta[1]))
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(start_epoch, end_epoch):
        iters = train(epoch, train_sets, G, Du, Dc, G_optim, Du_optim, Dc_optim, criterion, iters)
        test(epoch, G, train_sets)
        torch.save(G.state_dict(), os.path.join(args.checkpoints, 'gen'))
        torch.save(Du.state_dict(), os.path.join(args.checkpoints, 'dis_u'))
        torch.save(Dc.state_dict(), os.path.join(args.checkpoints, 'dis_c'))
        
def train(epoch, train_iter, G, Du, Dc, G_optim, Du_optim, Dc_optim, criterion, iters):
    G.train()
    Du.train()
    Dc.train()
    Tensor = torch.FloatTensor
    flag_real = torch.autograd.Variable(Tensor(batch_size).fill_(1.0), requires_grad=False).to(device)
    flag_fake = torch.autograd.Variable(Tensor(batch_size).fill_(0.0), requires_grad=False).to(device)
    
    for batch_ind, (inputs, targets) in enumerate(train_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        
        z = torch.randn(targets.size(0), 128, 1, 1).to(device)
        fake_u, fake_c, _, _ = G(z, targets)
        #print(fake_u.size(), fake_c.size())
        
        Du.zero_grad()
        dr_u = Du(inputs)
        df_u = Du(fake_u.detach())
        dr_loss_u = criterion(dr_u, flag_real[:targets.size(0)])
        df_loss_u = criterion(df_u, flag_fake[:targets.size(0)])
        d_loss_u = dr_loss_u + df_loss_u
        d_loss_u.backward()
        Du_optim.step()
        
        Dc.zero_grad()
        dr_c = Dc(inputs, targets)
        df_c = Dc(fake_c.detach(), targets)
        dr_loss_c = criterion(dr_c, flag_real[:targets.size(0)])
        df_loss_c = criterion(df_c, flag_fake[:targets.size(0)])
        d_loss_c = dr_loss_c + df_loss_c
        d_loss_c.backward()
        Dc_optim.step()
        
        G.zero_grad()
        z = torch.randn(targets.size(0), 128, 1, 1).to(device)
        fake_u, fake_c, mu, logvar = G(z, targets)
        kl_loss = KL_loss(mu, logvar)
        
        dg_u = Du(fake_u)
        dg_c = Dc(fake_c, targets)
        dg_loss_u = criterion(dg_u, flag_real[:targets.size(0)])
        dg_loss_c = criterion(dg_c, flag_real[:targets.size(0)])
        dg_loss = dg_loss_u + dg_loss_c + kl_loss
        dg_loss.backward()
        G_optim.step()
        
        iters += 1
        if batch_ind % 100 == 0:
            print('   Epoch: %d (%d iters) | Loss (G): %f | Loss (Du): %f | Loss (Dc): %f | D_KL: %f |'\
                % (epoch, iters, dg_loss.item(), d_loss_u.item(), d_loss_c.item(), kl_loss.item()))
            tb.add_scalars('Total loss', 
                {'dis_u': d_loss_u, 'dis_c': d_loss_c, 'gen': dg_loss}, 
                global_step=iters)
            tb.add_scalar('KL loss', kl_loss.item(), global_step=iters)
            
    return iters

def test(self, epoch, G, train_data):
    G.eval()
    for idx, (_, attr) in enumerate(train_data):
        if idx > 0:    break
        rand_idx = np.random.randint(len(attr))
        attr = attr.to(device)[rand_idx:rand_idx+1].repeat(100, 1)
        z = torch.randn(100, 128, 1, 1).type(torch.float32).to(device)
        with torch.no_grad():
            fake_img = G(z, attr)
        tb.add_images('Generated images %s' % attr.data.cpu()[0], fake_img, global_step=epoch)
        
if __name__ == '__main__':
    main()