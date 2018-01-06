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

class NetG(nn.Module):
    def __init__(self, image_size, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(NetG, self).__init__()
        self.ngpu = ngpu
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != image_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential(
            # input is Z going into a convolution
            nn.ConvTranspose2d(nz, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(cngf),
            nn.ReLU(True),
        )
        i, csize, cndf = 3, 4, cngf

        while csize < image_size//2:
            main.add_module(str(i), nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(str(i+1), nn.BatchNorm2d(cngf//2))
            main.add_module(str(i+2), nn.ReLU(True))

            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for j in range(n_extra_layers):
            main.add_module(str(i), nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(str(i+1), nn.BatchNorm2d(cngf))
            main.add_module(str(i+2), nn.ReLU(True))
            i += 3      
        
        # state size: K x 4 x 4
        main.add_module(str(i), nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main
    
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)

    

class NetD(nn.Module):
    def __init__(self, image_size, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(NetD, self).__init__()
        self.ngpu = ngpu
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        main = nn.Sequential(
            # input is nc x image_size x image_size
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        i, csize, cndf = 2, image_size / 2, ndf

        # Extra layers
        for j in range(n_extra_layers):
            main.add_module(str(i), nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(str(i+1), nn.BatchNorm2d(cndf))
            main.add_module(str(i+2), nn.LeakyReLU(0.2, inplace=True))
            i += 3
        
        while csize > 4:
            in_feature = cndf
            out_feature = cndf * 2
            main.add_module(str(i), nn.Conv2d(in_feature, out_feature, 4, 2, 1, bias=False))
            main.add_module(str(i+1), nn.BatchNorm2d(out_feature))
            main.add_module(str(i+2), nn.LeakyReLU(0.2, inplace=True))

            i += 3
            cndf = cndf * 2
            csize = csize / 2
        
        # state size: K x 4 x 4
        main.add_module(str(i), nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main
    
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        output = output.mean(0)
        return output.view(1)