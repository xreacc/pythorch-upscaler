import torch
from torch import nn

class attn_block(nn.Module):
    def __init__(self, ch_in, act=nn.LeakyReLU()):
        super(attn_block, self).__init__()
        self.channels = 32

        self.conv1 = nn.Conv2d(ch_in, self.channels, 3,1,1)
        self.conv2 = nn.Conv2d(ch_in, self.channels, 5,1,2)

        self.attn_map = nn.Sequential(
            act,
            nn.Conv2d(self.channels, ch_in, 3, 1, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        attn_map = self.attn_map(x1+x2)
        return (x * attn_map) + x


class resblock(nn.Module):
    def __init__(self, ch_in):
        super(resblock, self).__init__()
        self.act = nn.LeakyReLU()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, 16, 3, 1, 1),
            self.act,
            nn.Conv2d(16, 32, 3, 1, 1),
            self.act,
            nn.Conv2d(32, ch_in, 3, 1, 1),
            self.act,
        )
    def forward(self, x): return self.block(x) + x

class unet(nn.Module):
    def __init__(self, ch_in):
        super(unet, self).__init__()
        self.act = nn.LeakyReLU()
        self.channels = 32
        self.d = nn.ModuleList(
                 [nn.Sequential(nn.Conv2d(ch_in, self.channels, 2, 2), self.act),
                  nn.Sequential(nn.Conv2d(self.channels, self.channels*2, 2, 2), self.act),
                  nn.Sequential(nn.Conv2d(self.channels*2, self.channels*3, 2, 2), self.act),
                  ])
        self.mid = nn.ModuleList(
                   [nn.Sequential(attn_block(self.channels*3), resblock(self.channels*3), nn.Conv2d(self.channels*3,self.channels*3,3,1,1),self.act),
                    nn.Sequential(attn_block(self.channels*2), resblock(self.channels*2), nn.Conv2d(self.channels*2,self.channels*2,3,1,1),self.act),
                    nn.Sequential(attn_block(self.channels), resblock(self.channels), nn.Conv2d(self.channels,self.channels,3,1,1),self.act),
                    ])
        self.u = nn.ModuleList(
                 [nn.Sequential(nn.ConvTranspose2d(self.channels*3, self.channels*2, 4,2,1),self.act),
                  nn.Sequential(nn.ConvTranspose2d(self.channels*2, self.channels, 4,2,1),self.act),
                  nn.Sequential(nn.ConvTranspose2d(self.channels, ch_in, 4,2,1),self.act),
                  ])
    def forward(self, x):
        d1 = self.d[0](x)
        d2 = self.d[1](d1)

        d3 = self.d[2](d2)
        d3 = self.mid[0](d3)

        u1 = self.u[0](d3)
        u1 = self.mid[1](u1) + d2

        u2 = self.u[1](u1)
        u2 = self.mid[2](u2) + d1

        u3 = self.u[2](u2)

        return u3 + x



class upscaler(nn.Module):
    def __init__(self):
        super(upscaler, self).__init__()
        self.act = nn.LeakyReLU()

        self.up = nn.Sequential(
            nn.Conv2d(3, 15, 3,1,1,groups=3),
            self.act,
            attn_block(15),
            resblock(15),
            resblock(15),
            nn.Conv2d(15, 15, 3, 1, 1),
            self.act,

            nn.ConvTranspose2d(15, 30, 4,2,1),
            self.act,
            nn.Conv2d(30, 30, 3,1,1,groups=3),
            self.act,
            attn_block(30),
            resblock(30),
            resblock(30),
            nn.Conv2d(30, 30, 3, 1, 1),
            self.act,

            nn.ConvTranspose2d(30, 60, 4,2,1),
            self.act,
            nn.Conv2d(60, 15, 3,1,1,groups=3),
            self.act,
            attn_block(15),
            resblock(15),
            resblock(15),
        )
        self.unets = nn.Sequential(*[unet(15) for _ in range(8)])
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(15, 30, 3, 1, 1, groups=3),
            self.act,
            nn.Conv2d(30, 3, 3, 1, 1, groups=3),
        )
    def forward(self, x):
        x = self.up(x)
        x = self.unets(x)
        x = self.conv_last(x)
        return x
