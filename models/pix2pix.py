import torch
import torch.nn as nn
from torch.autograd import Variable

from models.generators import Unet
from models.discriminators import NLayerDiscriminator
from util import ImagePool


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super().__init__()

        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None

        self.Tensor = tensor

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None

        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))

            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)

            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class Pix2Pix(object):
    def __init__(self,
                 input_nc, output_nc,
                 num_downs, ngf,
                 ndf, n_layers_D,
                 which_direction="AtoB",
                 norm=nn.BatchNorm2d, use_dropout=False, init_type='normal', no_lsgan=True, learning_rate=0.1,
                 pool_size=50, lambda_A=10.0, lambda_B=10.0):
        """

        :param input_nc:
        :param output_nc:
        :param num_downs:
        :param ngf:
        :param ndf:
        :param n_layers_D:
        :param which_direction:
        :param norm:
        :param use_dropout:
        :param init_type:
        :param no_lsgan: do not use least square GAN, if false, use vanilla GAN
        :param learning_rate:
        :param pool_size:
        :param lambda_A:
        :param lambda_B:
        """
        super().__init__()
        self.which_direction = which_direction
        self.fake_AB_pool = ImagePool(pool_size)
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        self.generator = Unet(input_nc=input_nc, output_nc=output_nc, num_downs=num_downs, ngf=ngf,
                              norm_layer=norm, use_dropout=use_dropout)
        self.discriminator = NLayerDiscriminator(input_nc=input_nc+output_nc, ndf=ndf, n_layers=n_layers_D, use_sigmoid=no_lsgan)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        self.criterionGAN = GANLoss(use_lsgan=not no_lsgan, tensor=torch.FloatTensor)
        self.criterionL1 = torch.nn.L1Loss()

    def set_input(self, input):
        AtoB = self.which_direction == "AtoB"
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']

        self.input_A = input_A.cuda()
        self.input_B = input_B.cuda()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.generator(self.real_A)
        self.real_B = Variable(self.input_B)

    def backward_D(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        # fake_B is results of generator, so must be removed from computation graph during training discriminator
        pred_fake = self.discriminator(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.discriminator(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.discriminator(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)