import itertools
import torch
import numpy as np
import os
from Lib.utils.generic.generic_utils import start, stop
from Lib.Nets.utils.arch.arch import Generator, Discriminator
from Lib.Nets.utils.generic.init_weights import init_weights
from Lib.Nets.utils.generic.DecayLR import DecayLR
from Lib.Nets.utils.generic.image2tensorboard import display_input, reconstruct_tile
from tqdm import tqdm
from Lib.utils.Logger.Logger import Logger
from Lib.Nets.utils.generic.generic_training import set_requires_grad, breaker, cal_gp, List, drop_channel
from Lib.Nets.utils.generic.ReplyBuffer import ReplayBuffer
from Lib.Nets.utils.generic.trainSN import trainSN
import pickle as pkl


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file implements cycle consistent adversarial transcoder
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Cycle_AT:
    """
    This class implements all the variables, functions, methods, required to deploy Cycle-GAN [ref]
    """
    def __init__(self, opt, device):
        """
        define networks (both Generators and discriminators)
        S -> SAR
        O -> Optical
        real data label is 1, fake data label is 0.
        Cycle GAN init function
        :param opt: opt contain all the variables that are configurable when launching the script, check the
        folder: Lib/Nets/utils/config/ there are three scripts which are used to configure networks
        :param device: cuda device
        :return:
        """
        # images placeholders
        self.real_S = None
        self.real_O = None
        self.fake_O = None
        self.rec_S = None
        self.fake_S = None
        self.rec_O = None
        self.fake_or_real_S = None
        self.fake_or_real_O = None
        self.idt_S = None
        self.idt_O = None
        self.name = None
        # cost function values
        self.loss_idt_S2O = None
        self.loss_idt_O2S = None
        self.loss_adv_G_S2O = None
        self.loss_adv_G_O2S = None
        self.loss_cycle_S2O = None
        self.loss_cycle_O2S = None
        self.loss_G = None
        self.loss_D_S_real = None
        self.loss_D_S_fake = None
        self.loss_D_S_gp = None
        self.loss_D_S = None
        self.loss_D_O_real = None
        self.loss_D_O_fake = None
        self.loss_D_O_gp = None
        self.loss_D_O = None

        # cost function buffers
        self.loss_real_buff_S = List(opt.buff_dim)
        self.loss_buff_S = List(opt.buff_dim)
        self.loss_real_buff_O = List(opt.buff_dim)
        self.loss_buff_O = List(opt.buff_dim)
        self.D_ratio_O = opt.D_training_ratio
        self.D_ratio_S = opt.D_training_ratio
        self.drop_O = False

        # general
        self.device = device
        self.opt = opt
        self.Logger = Logger(self.opt.mode)
        self.trans_o = None
        self.trans_s = None
        self.trans_o_eval = None
        self.trans_s_eval = None
        self.sar_c_vis = min(self.opt.sar_c, 3)
        self.opt_c_vis = min(self.opt.optical_c, 3)
        self.posx_train = pkl.load(open(os.path.join(opt.data_dir_train, 'posx.pkl'), "rb"))
        self.posy_train = pkl.load(open(os.path.join(opt.data_dir_train, 'posy.pkl'), "rb"))
        self.posx_test = pkl.load(open(os.path.join(opt.data_dir_test, 'posx.pkl'), "rb"))
        self.posy_test = pkl.load(open(os.path.join(opt.data_dir_test, 'posy.pkl'), "rb"))
        self.flag = False
        # Generators
        self.netG_S2O = Generator(self.opt.sar_c, self.opt.optical_c, self.opt.dropout, self.opt.bias).to(self.device)
        self.netG_O2S = Generator(self.opt.optical_c, self.opt.sar_c, self.opt.dropout, self.opt.bias).to(self.device)

        if self.opt.mode == "train":
            print('Mode -> train')
            if self.opt.conditioned or self.opt.dropping:
                temp1 = self.opt.sar_c + self.opt.optical_c
                temp2 = temp1
            else:
                temp1 = self.opt.sar_c
                temp2 = self.opt.optical_c
            self.netD_S = Discriminator(temp1, self.opt.bias).to(self.device)
            self.netD_O = Discriminator(temp2, self.opt.bias).to(self.device)

            f = open(os.path.join(self.opt.log_dir, 'net_arch.txt'), 'a')
            f.write("Discriminator:\n{}\n".format(self.netD_S))
            f.write("Discriminator:\n{}\n".format(self.netG_S2O))
            f.close()

            set_requires_grad(self.netG_S2O, True)
            set_requires_grad(self.netG_O2S, True)
            set_requires_grad(self.netD_S, True)
            set_requires_grad(self.netD_O, True)
            # init weights
            init_weights(self.netG_S2O, self.opt.init_type, self.opt.init_gain)
            init_weights(self.netG_O2S, self.opt.init_type, self.opt.init_gain)
            init_weights(self.netD_S, self.opt.init_type, self.opt.init_gain)
            init_weights(self.netD_O, self.opt.init_type, self.opt.init_gain)
            # only works when input and output images have the same number of channels
            if self.opt.lambda_identity > 0.0:
                assert(self.opt.sar_c == self.opt.optical_c), 'SAR has {}, Optical has {}'\
                    .format(self.opt.sar_c, self.opt.optical_c)
            # create image buffer to store previously generated images
            self.fake_S_pool = ReplayBuffer(self.opt.pool_size)
            self.fake_O_pool = ReplayBuffer(self.opt.pool_size)
            # define loss functions
            # MSE = lsgan, BCE = vanilla, niente = wgan
            if self.opt.loss_type == 'lsgan':
                self.criterionGAN = torch.nn.MSELoss().to(self.device)
            elif self.opt.loss_type == 'wgan':
                self.criterionGAN = None
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            # initialize optimizers
            self.G_net_chain = itertools.chain(self.netG_S2O.parameters(), self.netG_O2S.parameters())
            self.optimizer_G = torch.optim.Adam(self.G_net_chain, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D_S = torch.optim.Adam(self.netD_S.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D_O = torch.optim.Adam(self.netD_O.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            # instantiate the method step of the class decaylr
            self.lr_lambda = DecayLR(self.opt.tot_epochs, self.opt.start_from_epoch, self.opt.decay_epochs).step
            # initialise networks scheduler passing the function above
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)
            # [SETUP] TODO: merge D and optimizers
            self.lr_scheduler_D_S = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_S, lr_lambda=self.lr_lambda)
            self.lr_scheduler_D_O = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_O, lr_lambda=self.lr_lambda)
            if self.opt.restoring_rep_path is not None:
                file = os.path.join(self.opt.restoring_rep_path,
                                    'checkpoint_epoch_' + str(self.opt.start_from_epoch) + '.pt')
                self.load(file)
        elif self.opt.mode == "eval":
            #self.sar_c_vis = self.opt.sar_c
            #self.opt_c_vis = self.opt.optical_c
            print('Mode -> eval')
            set_requires_grad(self.netG_S2O, False)
            set_requires_grad(self.netG_O2S, False)
            file = os.path.join(self.opt.restoring_rep_path,
                                'checkpoint_epoch_' + str(self.opt.start_from_epoch) + '.pt')
            self.load(file)

    def set_input(self, data):
        """
        Unpack input data from the dataloader
        :param data:
        :return:
        """
        self.real_S = data['radar'].to(self.device)
        self.real_O = data['rgb'].to(self.device)
        self.name = data['name']

    def forward(self, var_name_o, var_name_s):
        """
        Run forward pass
        :return:
        """
        self.fake_O = self.netG_S2O(self.real_S)  # G_S(S)
        self.fake_S = self.netG_O2S(self.real_O)  # G_O(O)
        if self.opt.mode == "train":
            self.rec_S = self.netG_O2S(self.fake_O)  # G_O(G_S(S))
            self.rec_O = self.netG_S2O(self.fake_S)  # G_S(G_O(O))
        if self.flag:
            name = self.name
            img_o = self.fake_O.cpu().detach()
            img_s = self.fake_S.cpu().detach()
            for i, n in enumerate(name):
                temp = int(n.split('_')[1]) - 1
                getattr(self, var_name_o)[temp] = img_o[i, 0:self.opt_c_vis]
                getattr(self, var_name_s)[temp] = img_s[i, 0:self.sar_c_vis]


    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator
        Also call loss_D.backward() to calculate the gradients.
        :param netD: the discriminator D
        :param real: real images
        :param fake: images generated by a generator
        :return: Return the discriminator loss
        """
        # Real
        pred_real = netD(real)
        # Fake
        pred_fake = netD(fake.detach())
        if self.opt.loss_type == 'lsgan':
            loss_D_real = self.criterionGAN(pred_real, netD.real_label.expand_as(pred_real))
            loss_D_fake = self.criterionGAN(pred_fake, netD.fake_label.expand_as(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            temp = None
        elif self.opt.loss_type == 'wgan':
            gradient_penalty, _ = cal_gp(netD, real, fake, self.device, lambda_gp=self.opt.lambda_gp)
            gradient_penalty.backward(retain_graph=True)
            temp = gradient_penalty.item()
            loss_D_real = - pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
        return loss_D_real.item(), loss_D_fake.item(), loss_D.item(), temp

    def backward_D_S(self):
        """
        Calculate GAN loss for discriminator D_S
        :return:
        """
        # fake_O is a batch of fake image randomly sampled form last 50 generated images
        if self.opt.conditioned:
            temp1 = torch.cat((self.real_O, self.fake_S), 1)
            real = torch.cat((self.real_O, self.real_S), 1)
            fake_S = self.fake_S_pool.push_and_pop(temp1, self.opt.pool_prc_S)
            self.loss_D_S_real, self.loss_D_S_fake, self.loss_D_S, self.loss_D_S_gp = \
                self.backward_D_basic(self.netD_S, real, fake_S)
        elif self.opt.dropping:
            fake_S_drop = drop_channel(self.fake_S, p_th=(1-self.opt.pool_prc_S))
            real_O_drop = drop_channel(self.real_O, p_th=(1-self.opt.pool_prc_S))
            real_S_drop = drop_channel(self.real_S, p_th=(1-self.opt.pool_prc_S))
            #real_O_drop = self.real_O
            #real_S_drop = self.real_S
            temp1 = torch.cat((real_O_drop, fake_S_drop), 1)
            real = torch.cat((real_O_drop, real_S_drop), 1)
            fake_S = self.fake_S_pool.push_and_pop(temp1, self.opt.pool_prc_S)
            self.loss_D_S_real, self.loss_D_S_fake, self.loss_D_S, self.loss_D_S_gp = \
                self.backward_D_basic(self.netD_S, real, fake_S)
        else:
            fake_S = self.fake_S_pool.push_and_pop(self.fake_S, self.opt.pool_prc_S)
            #del
            #fake_S_drop = drop_channel(self.fake_S, p_th=(1-self.opt.pool_prc_S))
            #fake_S = self.fake_S_pool.push_and_pop(fake_S_drop, self.opt.pool_prc_S)
            self.loss_D_S_real, self.loss_D_S_fake, self.loss_D_S, self.loss_D_S_gp = \
                self.backward_D_basic(self.netD_S, self.real_S, fake_S)

    def backward_D_O(self):
        """
        Calculate GAN loss for discriminator D_O
        :return:
        """
        # fake_S is a batch of fake image randomly sampled form last 50 generated images
        if self.opt.conditioned:
            temp = torch.cat((self.real_S, self.fake_O), 1)
            real = torch.cat((self.real_S, self.real_O), 1)
            fake_O = self.fake_O_pool.push_and_pop(temp, self.opt.pool_prc_O)
            self.loss_D_O_real, self.loss_D_O_fake, self.loss_D_O, self.loss_D_O_gp = \
                self.backward_D_basic(self.netD_O, real, fake_O)
        elif self.opt.dropping:
            fake_O_drop = drop_channel(self.fake_O, p_th=(1-self.opt.pool_prc_O))
            real_S_drop = drop_channel(self.real_S, p_th=(1-self.opt.pool_prc_O))
            real_O_drop = drop_channel(self.real_O, p_th=(1-self.opt.pool_prc_O))
            #real_S_drop = self.real_S
            #real_O_drop = self.real_O
            temp = torch.cat((real_S_drop, fake_O_drop), 1)
            real = torch.cat((real_S_drop, real_O_drop), 1)
            fake_O = self.fake_O_pool.push_and_pop(temp, self.opt.pool_prc_O)
            self.loss_D_O_real, self.loss_D_O_fake, self.loss_D_O, self.loss_D_O_gp = \
                self.backward_D_basic(self.netD_O, real, fake_O)
        else:
            if self.drop_O:
                fake_O_drop = drop_channel(self.fake_O, p_th=(self.opt.drop_prc))
                real_O_drop = drop_channel(self.real_O, p_th=(self.opt.drop_prc))
                fake_O = self.fake_O_pool.push_and_pop(fake_O_drop, self.opt.pool_prc_O)
                self.loss_D_O_real, self.loss_D_O_fake, self.loss_D_O, self.loss_D_O_gp = \
                    self.backward_D_basic(self.netD_O, real_O_drop, fake_O)
            else:
                fake_O = self.fake_O_pool.push_and_pop(self.fake_O, self.opt.pool_prc_O)
                #del
                #fake_O_drop = drop_channel(self.fake_O, p_th=(1 - self.opt.pool_prc_O))
                #fake_O = self.fake_O_pool.push_and_pop(fake_O_drop, self.opt.pool_prc_O)
                self.loss_D_O_real, self.loss_D_O_fake, self.loss_D_O, self.loss_D_O_gp = \
                    self.backward_D_basic(self.netD_O, self.real_O, fake_O)

    def backward_G(self):
        """
        Calculate the loss for generators G_S and G_O
        :return:
        """
        # Identity loss
        if self.opt.lambda_identity > 0:
            # G_S should be identity if real_O is fed: ||G_S2O(O) - O||
            self.idt_S = self.netG_S2O(self.real_O)
            self.loss_idt_S2O = self.criterionIdt(self.idt_S, self.real_O) * self.opt.lambda_O * self.opt.lambda_identity
            # G_O should be identity if real_S is fed: ||G_O2S(S) - S||
            self.idt_O = self.netG_O2S(self.real_S)
            self.loss_idt_O2S = self.criterionIdt(self.idt_O, self.real_S) * self.opt.lambda_S * self.opt.lambda_identity
        else:
            self.loss_idt_S2O = 0
            self.loss_idt_O2S = 0

        # GAN loss D_S(G_S2O(S))
        # self.netD_S.real_label it is only a parameter present in every discriminator instance
        if self.opt.conditioned or self.opt.dropping:
            temp = torch.cat((self.real_O, self.fake_S), 1)
            self.fake_or_real_S = self.netD_S(temp)
        else:
            self.fake_or_real_S = self.netD_S(self.fake_S)
        if self.opt.loss_type == 'lsgan':
            self.loss_adv_G_S2O = self.criterionGAN(self.fake_or_real_S, self.netD_S.real_label.expand_as(self.fake_or_real_S))
        elif self.opt.loss_type == 'wgan':
            self.loss_adv_G_S2O = - self.fake_or_real_S.mean()
        # GAN loss D_O(G_O2S(O))

        if self.opt.conditioned or self.opt.dropping:
            temp = torch.cat((self.real_S, self.fake_O), 1)
            self.fake_or_real_O = self.netD_O(temp)
        else:
            self.fake_or_real_O = self.netD_O(self.fake_O)
        if self.opt.loss_type == 'lsgan':
            self.loss_adv_G_O2S = self.criterionGAN(self.fake_or_real_O, self.netD_O.real_label.expand_as(self.fake_or_real_O))
        elif self.opt.loss_type == 'wgan':
            self.loss_adv_G_O2S = - self.fake_or_real_O.mean()
        # Forward cycle loss || G_O(G_S2O(S)) - S||
        self.loss_cycle_S2O = self.criterionCycle(self.rec_S, self.real_S) * self.opt.lambda_S
        # Backward cycle loss || G_S2O(G_O2S(O)) - O||
        self.loss_cycle_O2S = self.criterionCycle(self.rec_O, self.real_O) * self.opt.lambda_O
        # combined loss and calculate gradients
        self.loss_G = self.loss_adv_G_S2O + self.loss_adv_G_O2S + \
                      self.loss_cycle_S2O + self.loss_cycle_O2S
        self.loss_G.backward()

    def update_learning_rate(self):
        """
        This function is called to request a step to the scheduler so that to update the learning rate
        :return:
        """
        old_lr_G = self.optimizer_G.param_groups[0]['lr']
        old_lr_D_O = self.optimizer_D_O.param_groups[0]['lr']
        old_lr_D_S = self.optimizer_D_S.param_groups[0]['lr']

        self.lr_scheduler_G.step()
        self.lr_scheduler_D_S.step()
        self.lr_scheduler_D_O.step()

        lr_G = self.optimizer_G.param_groups[0]['lr']
        lr_D_O = self.optimizer_D_O.param_groups[0]['lr']
        lr_D_S = self.optimizer_D_S.param_groups[0]['lr']

        print('learning rate %.7f -> %.7f' % (old_lr_G, lr_G))
        print('learning rate %.7f -> %.7f' % (old_lr_D_O, lr_D_O))
        print('learning rate %.7f -> %.7f' % (old_lr_D_S, lr_D_S))

    def optimize_GAN(self, step):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        :return:
        """
        # compute fake images and reconstruction images.
        self.forward('trans_o', 'trans_s')
        # Ds require no gradients when optimizing Gs
        set_requires_grad(self.netD_S, False)
        set_requires_grad(self.netD_O, False)
        # set G_S and G_O's gradients to zero
        self.optimizer_G.zero_grad()
        # calculate gradients for G_S and G_O
        self.backward_G()
        # update G_S and G_O's weights
        self.optimizer_G.step()
        # Ds require gradients when optimizing them

        if step % self.D_ratio_S == 0:
            set_requires_grad(self.netD_S, True)
            # set D_S gradients to zero
            self.optimizer_D_S.zero_grad()
            # calculate gradients for D_S
            self.backward_D_S()
            # update D_S weights
            self.optimizer_D_S.step()

        if step % self.D_ratio_O == 0:
            set_requires_grad(self.netD_O, True)
            # set D_O gradients to zero
            self.optimizer_D_O.zero_grad()
            # calculate gradients for D_O
            self.backward_D_O()
            # update D_O weights
            self.optimizer_D_O.step()

        '''if step % 100 == 0:
            media = self.loss_buff_O.mean()
            if media > self.opt.th_high:
                self.drop_O = True
            else:
                self.drop_O = False'''

        if self.opt.pool:
            if self.loss_buff_O.mean() > self.opt.th_high:
                if self.D_ratio_O < self.opt.th_b_h_ratio:
                    self.D_ratio_O = self.D_ratio_O + 1
                if self.opt.pool_prc_O < self.opt.th_b_h_pool:
                    self.opt.pool_prc_O = self.opt.pool_prc_O + 0.1
            elif self.loss_buff_O.mean() < self.opt.th_low:
                if self.D_ratio_O > self.opt.th_b_l_ratio:
                    self.D_ratio_O = self.D_ratio_O - 1
                if self.opt.pool_prc_O > self.opt.th_b_l_pool:
                    self.opt.pool_prc_O = self.opt.pool_prc_O - 0.1

            if self.loss_buff_S.mean() > self.opt.th_high:
                if self.D_ratio_S < self.opt.th_b_h_ratio:
                    self.D_ratio_S = self.D_ratio_S + 1
                if self.opt.pool_prc_S < self.opt.th_b_h_pool:
                    self.opt.pool_prc_S = self.opt.pool_prc_S + 0.1
            elif self.loss_buff_S.mean() < self.opt.th_low:
                if self.D_ratio_S > self.opt.th_b_l_ratio:
                    self.D_ratio_S = self.D_ratio_S - 1
                if self.opt.pool_prc_S > self.opt.th_b_l_pool:
                    self.opt.pool_prc_S = self.opt.pool_prc_S - 0.1

    def save_model(self, epoch):
        """
        Save model parameters
        :param epoch: actual epoch
        :return:
        """
        out_file = os.path.join(self.opt.checkpoint_dir, 'checkpoint_epoch_' + str(epoch) + ".pt")
        # save model
        data = {"G_S2O": self.netG_S2O.state_dict(),
                "G_O2S": self.netG_O2S.state_dict(),
                "D_O": self.netD_O.state_dict(),
                "D_S": self.netD_S.state_dict(),
                "opt_G": self.optimizer_G.state_dict(),
                "opt_D_S": self.optimizer_D_S.state_dict(),
                "opt_D_O": self.optimizer_D_O.state_dict(),
                }
        torch.save(data, out_file)

    def load(self, file):
        """
        Restore model parameters
        :param file: file from where load parameters
        :return:
        """
        data = torch.load(file)
        if self.opt.mode == 'train':
            self.netG_S2O.load_state_dict(data['G_S2O'])
            self.netG_O2S.load_state_dict(data['G_O2S'])
            self.netD_O.load_state_dict(data['D_O'])
            self.netD_S.load_state_dict(data['D_S'])
            self.optimizer_G.load_state_dict(data['opt_G'])
            self.optimizer_D_S.load_state_dict(data['opt_D_S'])
            self.optimizer_D_O.load_state_dict(data['opt_D_O'])
        elif self.opt.mode == 'eval':
            self.netG_S2O.load_state_dict(data['G_S2O'])
            self.netG_O2S.load_state_dict(data['G_O2S'])

    def tb_add_step_loss_g(self, writer, global_step):
        """
        This function add G losses to tensorboard and store the value in the logger
        - loss_idt_S
        - loss_idt_O
        - loss_adv_G_S2O
        - loss_adv_G_O2S
        - loss_cycle_S
        - loss_cycle_O
        - loss_G
        :return: all losses and output of network
        """
        if self.opt.lambda_identity > 0:
            step_loss_G = {
                'loss_idt_S': self.loss_idt_S2O.item(),
                'loss_idt_O': self.loss_idt_O2S.item(),
                'loss_adv_G_S2O': self.loss_adv_G_S2O.item(),
                'loss_adv_G_O2S': self.loss_adv_G_O2S.item(),
                'loss_cycle_S': self.loss_cycle_S2O.item(),
                'loss_cycle_O': self.loss_cycle_O2S.item(),
                'loss_G': self.loss_G.item(),
            }
        else:
            step_loss_G = {
                'loss_adv_G_S2O': self.loss_adv_G_S2O.item(),
                'loss_adv_G_O2S': self.loss_adv_G_O2S.item(),
                'loss_cycle_S': self.loss_cycle_S2O.item(),
                'loss_cycle_O': self.loss_cycle_O2S.item(),
                'loss_G': self.loss_G.item(),
            }

        mean_loss = {
            'prc_S': self.opt.pool_prc_S,
            'prc_O': self.opt.pool_prc_O,
            'D_ratio_O': self.D_ratio_O,
            'D_ratio_S': self.D_ratio_S,
        }
        writer.add_scalars("Train/Generator", step_loss_G, global_step=global_step)
        writer.add_scalars("Train/mean", mean_loss, global_step=global_step)
        self.Logger.append_G(step_loss_G)

    def tb_add_step_loss_d_s(self, writer, global_step):
        """
        This function add D_S losses to tensorboard and store the value in the logger
        - loss_D_S_real
        - loss_D_S_fake
        - loss_D_S
        :return: all losses and output of network
        """
        if self.opt.loss_type == 'lsgan':
            step_loss_D_S = {
                'loss_D_S_real': self.loss_D_S_real,
                'loss_D_S_fake': self.loss_D_S_fake,
                'loss_D_S': self.loss_D_S,
            }
        elif self.opt.loss_type == 'wgan':
            step_loss_D_S = {
                'loss_D_S_real': self.loss_D_S_real,
                'loss_D_S_fake': self.loss_D_S_fake,
                'loss_D_S_gp': self.loss_D_S_gp,
                'loss_D_S': self.loss_D_S,
            }

        writer.add_scalars("Train/Discriminator_Sar", step_loss_D_S, global_step=global_step)
        self.Logger.append_D_S(step_loss_D_S)

    def tb_add_step_loss_d_o(self, writer, global_step):
        """
        This function add D_O losses to tensorboard and store the value in the logger
        - loss_D_O_real
        - loss_D_O_fake
        - loss_D_O
        :return: all losses and output of network
        """
        if self.opt.loss_type == 'lsgan':
            step_loss_D_O = {
                'loss_D_O_real': self.loss_D_O_real,
                'loss_D_O_fake': self.loss_D_O_fake,
                'loss_D_O': self.loss_D_O,
            }
        elif self.opt.loss_type == 'wgan':
            step_loss_D_O = {
                'loss_D_O_real': self.loss_D_O_real,
                'loss_D_O_fake': self.loss_D_O_fake,
                'loss_D_O_gp': self.loss_D_O_gp,
                'loss_D_O': self.loss_D_O,
            }
        writer.add_scalars("Train/Discriminator_Opt", step_loss_D_O, global_step=global_step)
        self.Logger.append_D_O(step_loss_D_O)

    def tb_add_step_images(self, writer=None, global_step=None):
        """
        Saves all net images to tendorboard
        :param writer: pointer to tb
        :param global_step: step for tb
        :return:
        """
        real_S_norm = display_input(self.real_S[:, 0:self.sar_c_vis, :, :], False)
        real_O_norm = display_input(self.real_O[:, 0:3, :, :], False)
        real = np.concatenate([real_S_norm, real_O_norm])

        fake_S_norm = display_input(self.fake_S[:, 0:self.sar_c_vis, :, :], False)
        fake_O_norm = display_input(self.fake_O[:, 0:3, :, :], False)
        fake = np.concatenate([fake_S_norm, fake_O_norm])

        if self.opt.mode == "train":
            rec_S_norm = display_input(self.rec_S[:, 0:self.sar_c_vis, :, :], False)
            rec_O_norm = display_input(self.rec_O[:, 0:3, :, :], False)
            rec = np.concatenate([rec_S_norm, rec_O_norm])

            if self.opt.lambda_identity > 0:
                idt_S_norm = display_input(self.idt_S[:, 0:self.sar_c_vis, :, :], False)
                idt_O_norm = display_input(self.idt_O[:, 0:3, :, :], False)
                idt = np.concatenate([idt_S_norm, idt_O_norm])
        # if the writer is not passed to the function instead of updating tb it returns the images,
        # this variant is useful to create tile
        if writer is None:
            return real_S_norm[0], real_O_norm[0], fake_S_norm[0], fake_O_norm[0]
        else:
            writer.add_images("1 - REAL - Real Radar - Real Optical", real, global_step=global_step)
            writer.add_images("2 - FAKE - Fake Radar G_O2S(O) - Fake Optical G_S2O(S)", fake, global_step=global_step)
        if self.opt.mode == "train":
            writer.add_images("3 - RECONSTRUCTED - Radar G_O2S(Fake_O) - Optical G_S2O(Fake_S)", rec, global_step=global_step)
            if self.opt.lambda_identity > 0.0:
                writer.add_images("4 - IDENTITY - Radar G_O2S(real_S) - Optical G_S2O(Fake_O)", idt, global_step=global_step)

    def tb_add_step_d_output(self, writer, global_step):
        """
        - fake_or_real_S
        - fake_or_real_O
        :return: all losses and output of network
        """
        # [6] TODO:  implement a way to add to tensorboard the output of the discriminators
        step_data_D_result = {
            'fake_or_real_S': self.fake_or_real_S,
            'fake_or_real_O': self.fake_or_real_O,
        }
        writer = writer
        global_step = global_step

        return step_data_D_result, writer, global_step

    def reset_dyn(self):
        self.opt.pool_prc_S = 0.7
        self.opt.pool_prc_O = 0.7
        self.D_ratio_O = self.opt.D_training_ratio
        self.D_ratio_S = self.opt.D_training_ratio

    def train(self, train_dataset, eval_dataset, writer=None):
        """
        Run the training for the required epochs
        :param train_dataset: dataset used to train the network
        :param eval_dataset: dataset used to eval the network
        :param writer: a tensorboard instance to track info
        :return:
        """
        self.trans_o = torch.zeros((len(train_dataset.dataset), self.opt_c_vis, self.opt.patch_size, self.opt.patch_size),
                                   dtype=torch.float32)
        self.trans_s = torch.zeros((len(train_dataset.dataset), self.sar_c_vis, self.opt.patch_size, self.opt.patch_size),
                                   dtype=torch.float32)
        epoch = self.opt.start_from_epoch
        global_step = epoch * len(train_dataset)
        for epoch in range(epoch, self.opt.tot_epochs):
            t = start()
            text_line = "=" * 30 + "EPOCH " + str(epoch) + "/" + str(self.opt.tot_epochs) + "=" * 30
            print(text_line)
            progress_bar = tqdm(enumerate(train_dataset), total=len(train_dataset))
            # Train for each patch in the
            for i, data in progress_bar:
                self.set_input(data)
                self.optimize_GAN(global_step)
                self.loss_buff_S.push_and_pop(self.loss_adv_G_S2O)
                self.loss_buff_O.push_and_pop(self.loss_adv_G_O2S)
                # write generator loss to tensorboard
                if global_step > 0 and global_step % self.opt.loss_log_freq == 0:
                    self.tb_add_step_loss_g(writer, global_step)
                    self.tb_add_step_loss_d_s(writer, global_step)
                    self.tb_add_step_loss_d_o(writer, global_step)
                    self.Logger.append_loss_step({"step": global_step})
                # if global_step > 0 and global_step % self.opt.images_log_freq == 0:
                    # self.tb_add_step_images(writer, global_step)
                global_step = global_step + 1

            self.update_learning_rate()
            #self.reset_dyn()
            if epoch >= 0 and epoch % self.opt.save_model_freq == 0:
                self.save_model(epoch)
                # reconstruct_tile('train_o', self.opt.patch_size, self.posx_train, self.posy_train, self.opt.tb_dir,
                #                  self.opt.train_size, epoch, self.trans_o)  # , parameter_path=par_path)
                # reconstruct_tile('train_s', self.opt.patch_size, self.posx_train, self.posy_train, self.opt.tb_dir,
                #                 self.opt.train_size, epoch, self.trans_s)  # , parameter_path=par_path)
                trainSN(self.opt, epoch, self.device)

            if epoch >= 0 and epoch % self.opt.images_log_freq == 0:
                self.eval(eval_dataset, epoch, self.posx_test, self.posy_test, 'test')
                self.eval(train_dataset, epoch, self.posx_train, self.posy_train, 'train')

            self.Logger.save_logger(self.opt.checkpoint_dir, '')
            # Epoch duration
            s = 'Epoch {} took'.format(epoch)
            stop(t, s)
            if breaker(self.opt, epoch):
                print('EXECUTION FORCED TO STOP AT {} EPOCHS'.format(epoch))
                break

    def eval(self, dataset=None, epoch=0, posx=None, posy=None, name=''):
        set_requires_grad(self.netG_S2O, False)
        self.flag = True
        self.trans_o_eval = torch.zeros((len(dataset.dataset), self.opt_c_vis, self.opt.patch_size, self.opt.patch_size),
                                        dtype=torch.float32)
        self.trans_s_eval = torch.zeros((len(dataset.dataset), self.sar_c_vis, self.opt.patch_size, self.opt.patch_size),
                                        dtype=torch.float32)
        progress_bar = tqdm(enumerate(dataset), total=len(dataset))
        # Train for each patch in the
        for i, data in progress_bar:
            self.set_input(data)
            self.forward('trans_o_eval', 'trans_s_eval')
        reconstruct_tile(name + '_o', self.opt.patch_size, posx, posy, self.opt.tb_dir, self.opt.train_size, epoch,
                         self.trans_o_eval)  # , parameter_path=par_path)
        reconstruct_tile(name + '_s', self.opt.patch_size, posx, posy, self.opt.tb_dir, self.opt.train_size, epoch,
                         self.trans_s_eval, rgb=False)  # , parameter_path=par_path)
        print('DONEDONEDONE')
        print('DONEDONEDONE')
        print('DONEDONEDONE')
        print('DONEDONEDONE')
        self.flag = False
        set_requires_grad(self.netG_S2O, True)
