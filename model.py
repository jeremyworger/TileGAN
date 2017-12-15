import torch
import torch.nn as nn
import torch.nn.parallel

def weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# DCGAN model, fully convolutional architecture
class NetG1(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf, n_extra_layers_g):
        super(NetG1, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
        )

        # Extra layers
        for t in range(n_extra_layers_g):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, ngf),
                            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ngf),
                            nn.BatchNorm2d(ngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, ngf),
                            nn.LeakyReLU(0.2, inplace=True))

        main.add_module('final_layer.deconv', 
        	             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)) # 5,3,1 for 96x96
        main.add_module('final_layer.tanh', 
        	             nn.Tanh())
            # state size. (nc) x 96 x 96

        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu)), 0
        else:
            output = self.main(input)
        return output

class NetD1(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf,  n_extra_layers_d):
        super(NetD1, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 5,3,1 for 96x96
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        # Extra layers
        for t in range(n_extra_layers_d):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, ndf * 8),
                            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ndf * 8),
                            nn.BatchNorm2d(ndf * 8))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, ndf * 8),
                            nn.LeakyReLU(0.2, inplace=True))


        main.add_module('final_layers.conv', nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        main.add_module('final_layers.sigmoid', nn.Sigmoid())
        # state size. 1 x 1 x 1
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        return output.view(-1, 1)