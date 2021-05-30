import torch
import numpy as np
import os
from Lib.Nets.utils.arch.arch import Generator, Discriminator
from Lib.Nets.utils.generic.init_weights import init_weights
from Lib.Nets.utils.generic.DecayLR import DecayLR
from Lib.Nets.utils.generic.image2tensorboard import display_input, reconstruct_tile
from tqdm import tqdm
from Lib.utils.Logger.Logger import Logger
from Lib.Nets.utils.generic.generic_training import set_requires_grad, breaker, cal_gp, List, drop_channel
from Lib.Nets.utils.generic.ReplyBuffer import ReplayBuffer
from Lib.utils.generic.generic_utils import start, stop
from Lib.Nets.utils.generic.trainSN import trainSN
import pickle as pkl


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file implements the conditional adversarial transcoder
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class CAT:
    """
    This class implements all the variables, functions, methods, required to deploy pix2pix [ref] so a conditional
    GAN trained in an adversarial fashion
    """
    def __init__(self, opt, device):
        """
        define networks (both Generators and discriminators)
        real data label is 1, fake data label is 0.
        :param opt: opt contain all the variables that are configurable when launching the script, check the
        folder: Lib/Nets/utils/config/ there are three scripts which are used to configure networks
        :param device: cuda device
        :return:
        """
        # images placeholders
        self.real_S = None
        self.label = None   # real optical
        self.name = None
        self.fake_O = None
        self.fake_or_real = None

        # cost function values
        self.loss_adv_G = None
        self.loss_direct_G = None
        self.loss_G = None
        self.loss_D_O_real = None
        self.loss_D_O_fake = None
        self.loss_D_O_gp = None
        self.loss_D_O = None

        # cost function buffers
        self.loss_real_buff = List(opt.buff_dim)
        self.loss_fake_buff = List(opt.buff_dim)
        self.D_ratio = opt.D_training_ratio

        # general
        self.device = device
        self.opt = opt
        self.Logger = Logger(self.opt.mode)
        self.trans = None
        self.trans_eval = None
        self.sar_c_vis = min(self.opt.sar_c, 3)
        self.opt_c_vis = min(self.opt.optical_c, 3)
        self.posx_train = pkl.load(open(os.path.join(opt.data_dir_train, 'posx.pkl'), "rb"))
        self.posy_train = pkl.load(open(os.path.join(opt.data_dir_train, 'posy.pkl'), "rb"))
        self.posx_test = pkl.load(open(os.path.join(opt.data_dir_test, 'posx.pkl'), "rb"))
        self.posy_test = pkl.load(open(os.path.join(opt.data_dir_test, 'posy.pkl'), "rb"))
        self.flag = False
        # Generator
        self.netG_S2O = Generator(self.opt.sar_c, self.opt.optical_c, self.opt.dropout, self.opt.bias).to(self.device)

        if self.opt.mode == "train":
            print('Mode -> train')
            self.netD_O = Discriminator(self.opt.sar_c + self.opt.optical_c, self.opt.bias).to(self.device)
            set_requires_grad(self.netG_S2O, True)
            set_requires_grad(self.netD_O, True)
            # init weights
            init_weights(self.netG_S2O, self.opt.init_type, self.opt.init_gain)
            init_weights(self.netD_O, self.opt.init_type, self.opt.init_gain)
            self.fake_O_pool = ReplayBuffer(self.opt.pool_size)
            # define loss functions
            if self.opt.loss_type == 'lsgan':
                self.criterion_adv = torch.nn.MSELoss().to(self.device)
            elif self.opt.loss_type == 'wgan':
                self.criterion_adv = None
            self.criterion_G = torch.nn.L1Loss().to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_S2O.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D_O = torch.optim.Adam(self.netD_O.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            # instantiate the method step of the class decaylr
            self.lr_lambda = DecayLR(self.opt.tot_epochs, self.opt.start_from_epoch, self.opt.decay_epochs).step
            # initialise networks scheduler passing the function above
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)
            self.lr_scheduler_D_O = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_O, lr_lambda=self.lr_lambda)
            if self.opt.restoring_rep_path is not None:
                file = os.path.join(self.opt.restoring_rep_path,
                                    'checkpoint_epoch_' + str(self.opt.start_from_epoch) + '.pt')
                self.load(file)
        elif self.opt.mode == "eval":
            print('Mode -> eval')
            set_requires_grad(self.netG_S2O, False)
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
        self.label = data['rgb'].to(self.device)
        self.name = data['name']

    def forward(self, var_name):
        """
        Run forward pass
        :return:
        """
        self.fake_O = self.netG_S2O(self.real_S)  # G_S(S)
        if self.flag:
            name = self.name
            img = self.fake_O.cpu().detach()
            for i, n in enumerate(name):
                temp = int(n.split('_')[1]) - 1
                getattr(self, var_name)[temp] = img[i, 0:3]

    def backward_D(self):
        """
        Calculate GAN loss for the discriminator
        Also call loss_D.backward() to calculate the gradients.
        Conditional GANs needs to feed both input and output to the discriminator
        with detach stop the backpropagation to the generator
        [SETUP] TODO:There is a GitLab version which only here uses the pool
        :return: NA
        """
        # Real
        if self.opt.dropping:
            label_drop = drop_channel(self.fake_O)
            real_S_drop = drop_channel(self.real_S)
            real_SO = torch.cat((real_S_drop, label_drop), 1)
        else:
            real_SO = torch.cat((self.real_S, self.label), 1)
        if self.opt.pool:
            real_SO = self.fake_O_pool.push_and_pop(real_SO, self.opt.pool_prc_O)
        pred_real = self.netD_O(real_SO)
        # Fake

        if self.opt.dropping:
            fake_O_drop = drop_channel(self.fake_O)
            real_S_drop = drop_channel(self.real_S)
            fake_SO = torch.cat((real_S_drop, fake_O_drop), 1)
        else:
            fake_SO = torch.cat((self.real_S, self.fake_O), 1)
        if self.opt.pool:
            fake_SO = self.fake_O_pool.push_and_pop(fake_SO, self.opt.pool_prc_O)
        pred_fake = self.netD_O(fake_SO.detach())
        # Combined loss and calculate gradients
        if self.opt.loss_type == 'lsgan':
            loss_D_real = self.criterion_adv(pred_real, self.netD_O.real_label.expand_as(pred_real))
            loss_D_fake = self.criterion_adv(pred_fake, self.netD_O.fake_label.expand_as(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
        elif self.opt.loss_type == 'wgan':
            gradient_penalty, _ = cal_gp(self.netD_O, real_SO, fake_SO, self.device, lambda_gp=self.opt.lambda_gp)
            gradient_penalty.backward(retain_graph=True)
            loss_D_real = - pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
            self.loss_D_O_gp = gradient_penalty.item()

        self.loss_D_O_real = loss_D_real.item()
        self.loss_D_O_fake = loss_D_fake.item()
        self.loss_D_O = loss_D.item()

    def backward_G(self):
        """
        Calculate the loss for generators
        :return:
        """
        # self.netD_O.real_label is a discriminator parameter
        fake_SO = torch.cat((self.real_S, self.fake_O), 1)
        self.fake_or_real = self.netD_O(fake_SO)
        if self.opt.loss_type == 'lsgan':
            self.loss_adv_G = self.criterion_adv(self.fake_or_real, self.netD_O.real_label.expand_as(self.fake_or_real))
        elif self.opt.loss_type == 'wgan':
            self.loss_adv_G = - self.fake_or_real.mean()
        # Forward cycle loss ||G_S2O(S) - O||
        self.loss_direct_G = self.criterion_G(self.fake_O, self.label) * self.opt.lambda_A
        # combined loss and calculate gradients
        self.loss_G = self.loss_adv_G + self.loss_direct_G
        self.loss_G.backward()

    def update_learning_rate(self):
        """
        This function is called to request a step to the scheduler so that to update the learning rate
        :return:
        """
        # update scheduler
        old_lr_G = self.optimizer_G.param_groups[0]['lr']
        old_lr_D_O = self.optimizer_D_O.param_groups[0]['lr']

        self.lr_scheduler_G.step()
        self.lr_scheduler_D_O.step()

        lr_G = self.optimizer_G.param_groups[0]['lr']
        lr_D_O = self.optimizer_D_O.param_groups[0]['lr']

        print('learning rate %.7f -> %.7f' % (old_lr_G, lr_G))
        print('learning rate %.7f -> %.7f' % (old_lr_D_O, lr_D_O))

    def optimize(self, step):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        :return:
        """
        # compute fake images and reconstruction images.
        self.forward('trans')
        if step % self.D_ratio == 0:
            # Ds require gradients when optimizing them
            set_requires_grad(self.netD_O, True)
            # set D_O gradients to zero
            self.optimizer_D_O.zero_grad()
            # calculate gradients for D_O
            self.backward_D()
            # update D_O weights
            self.optimizer_D_O.step()
        # Ds require no gradients when optimizing Gs
        set_requires_grad(self.netD_O, False)
        # set G_S and G_O's gradients to zero
        self.optimizer_G.zero_grad()
        # calculate gradients for G_S and G_O
        self.backward_G()
        # update G_S and G_O's weights
        self.optimizer_G.step()

        '''if self.loss_real_buff.mean() < self.opt.th_low and self.loss_fake_buff.mean() < self.opt.th_low:
            if self.D_ratio < 10:
                self.D_ratio = self.D_ratio + 1
        elif self.loss_real_buff.mean() > self.opt.th_high and self.loss_fake_buff.mean() > self.opt.th_high:
            if self.D_ratio > 1:
                self.D_ratio = self.D_ratio - 1
        else:
            self.D_ratio = self.opt.D_training_ratio'''

    def save_model(self, epoch):
        """
        Save model parameters
        :param epoch: actual epoch
        :return:
        """
        out_file = os.path.join(self.opt.checkpoint_dir, 'checkpoint_epoch_' + str(epoch) + ".pt")
        data = {"G_S2O": self.netG_S2O.state_dict(),
                "D_O": self.netD_O.state_dict(),
                "opt_G": self.optimizer_G.state_dict(),
                "opt_D_O": self.optimizer_D_O.state_dict(),
                }
        torch.save(data, out_file)

    def load(self, file):
        """
        restore model parameters
        :param file: file from where load parameters
        :return:
        """
        data = torch.load(file)
        if self.opt.mode == 'train':
            self.netG_S2O.load_state_dict(data['G_S2O'])
            self.netD_O.load_state_dict(data['D_O'])
            self.optimizer_G.load_state_dict(data['opt_G'])
            self.optimizer_D_O.load_state_dict(data['opt_D_O'])
        elif self.opt.mode == 'eval':
            self.netG_S2O.load_state_dict(data['G_S2O'])

    def tb_add_step_loss_g(self, writer, global_step):
        """
        This function add G losses to tensorboard and store the value in the logger
        - loss_adv
        - loss_direct
        - loss_G
        :return: all losses and output of network
        """
        step_loss_G = {
            'loss_adv': self.loss_adv_G.item(),
            'loss_direct': self.loss_direct_G.item(),
            'loss_G': self.loss_G.item(),
        }
        writer.add_scalars("Train/Generator", step_loss_G, global_step=global_step)
        self.Logger.append_G(step_loss_G)

    def tb_add_step_loss_d_o(self, writer, global_step):
        """
        This function add D losses to tensorboard and store the value in the logger
        - loss_D_O_real
        - loss_D_O_fake
        - loss_D_O_gp
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
        writer.add_scalars("Train/Discriminator", step_loss_D_O, global_step=global_step)
        self.Logger.append_D_O(step_loss_D_O)

    def tb_add_step_images(self, writer=None, global_step=None):
        """
        Saves all net images to tensorboard
        :param writer: pointer to tb
        :param global_step: step for tb
        :return:
        """
        real_S_norm = display_input(self.real_S[:, 0:self.sar_c_vis, :, :], False)
        real_O_norm = display_input(self.label[:, 0:3, :, :], False)
        fake_O_norm = display_input(self.fake_O[:, 0:3, :, :], False)
        real = np.concatenate([real_S_norm, real_O_norm, fake_O_norm])
        # if the writer is not passed to the function instead of updating tb it returns the images,
        # this variant is useful to create tile
        if writer is None:
            return real_S_norm[0], real_O_norm[0], fake_O_norm[0]
        else:
            writer.add_images("Real Radar - Real Optical - Fake Optical", real, global_step=global_step)

    def tb_add_step_d_output(self, writer, global_step):
        """
        - fake_or_real_S
        - fake_or_real_O
        :return: all losses and output of network
        """
        # [6] TODO:  implement a way to add to tensorboard the output of the discriminators
        step_data_D_result = {
            'fake_or_real_O': self.fake_or_real,
        }
        writer = writer
        global_step = global_step

        return step_data_D_result, writer, global_step

    def train(self, train_dataset, eval_dataset, writer):
        """
        Run the training for the required epochs
        :param train_dataset: dataset used to train the network
        :param eval_dataset: dataset used to eval the network
        :param writer: a tensorboard instance to track info
        :return:
        """
        self.trans = torch.zeros((len(train_dataset.dataset), self.sar_c_vis, self.opt.patch_size, self.opt.patch_size),
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
                self.optimize(global_step)
                self.loss_real_buff.push_and_pop(self.loss_D_O_real)
                self.loss_fake_buff.push_and_pop(self.loss_D_O_fake)

                # write generator loss to tensorboard
                if global_step > 0 and global_step % self.opt.loss_log_freq == 0:
                    self.tb_add_step_loss_g(writer, global_step)
                    self.tb_add_step_loss_d_o(writer, global_step)
                    self.Logger.append_loss_step({"step": global_step})
                global_step = global_step + 1

            self.update_learning_rate()
            if epoch >= 0 and epoch % self.opt.save_model_freq == 0:
                self.save_model(epoch)
                # torch.save(self.trans, os.path.join(self.opt.tb_dir, str(epoch) + '.pt'))
                # reconstruct_tile('train', self.opt.patch_size, self.posx_train, self.posy_train, self.opt.tb_dir,
                                 # self.opt.train_size, epoch, self.trans)  # , parameter_path=par_path)
                if epoch >= 0 and epoch % self.opt.images_log_freq == 0:
                    self.eval(train_dataset, epoch, self.posx_train, self.posy_train, 'train')
                    self.eval(eval_dataset, epoch, self.posx_test, self.posy_test, 'test')
                trainSN(self.opt, epoch, self.device)

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
        self.trans_eval = torch.zeros((len(dataset.dataset), self.opt_c_vis, self.opt.patch_size, self.opt.patch_size),
                                      dtype=torch.float32)
        progress_bar = tqdm(enumerate(dataset), total=len(dataset))
        # Train for each patch in the
        for i, data in progress_bar:
            self.set_input(data)
            self.forward('trans_eval')
        reconstruct_tile(name, self.opt.patch_size, posx, posy, self.opt.tb_dir, self.opt.test_size, epoch, self.trans_eval)
        # , parameter_path=par_path)
        self.flag = False
        set_requires_grad(self.netG_S2O, True)

    def exp(self, dataset):
        self.flag = False
        progress_bar = tqdm(enumerate(dataset), total=len(dataset))
        for i, data in progress_bar:
            self.set_input(data)
            self.forward('trans_eval')
            fake_SO = torch.cat((self.real_S, self.fake_O), 1)
            self.fake_or_real = self.netD_O(fake_SO)
            #print(self.fake_or_real)
            self.loss_adv_G = self.criterion_adv(self.netD_O.real_label.expand_as(self.fake_or_real), self.netD_O.fake_label.expand_as(self.fake_or_real))
            #print(self.loss_adv_G)
            print(self.fake_or_real.mean())
