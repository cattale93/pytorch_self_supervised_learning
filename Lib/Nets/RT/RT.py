import torch
import numpy as np
import os
from Lib.Nets.utils.arch.arch import Generator
from Lib.Nets.utils.generic.init_weights import init_weights
from Lib.Nets.utils.generic.DecayLR import DecayLR
from Lib.Nets.utils.generic.image2tensorboard import display_input, reconstruct_tile
from tqdm import tqdm
from Lib.utils.Logger.Logger import Logger
from Lib.Nets.utils.generic.generic_training import set_requires_grad, breaker
from Lib.Nets.utils.generic.trainSN import trainSN
from Lib.utils.generic.generic_utils import start, stop
import pickle as pkl


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file implements the regressive transcoder
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class RT:
    """
    This class implements all the variables, functions, methods, required to deploy a very simple conditional GAN
    trained only based on a regression loss function
    """

    def __init__(self, opt, device):
        """
        :param opt: code options
        :param device: cuda device
        :return:
        """
        # images placeholders
        self.real_S = None
        self.fake_O = None
        self.label = None  # real_O (rgb)
        self.name = None

        # cost function values
        self.loss = None

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
        # Define generator
        self.netG_S2O = Generator(self.opt.sar_c, self.opt.optical_c, self.opt.dropout, self.opt.bias).to(self.device)

        if self.opt.mode == "train":
            print('Mode -> train')
            set_requires_grad(self.netG_S2O, True)
            # init weights
            init_weights(self.netG_S2O, self.opt.init_type, self.opt.init_gain)
            # define loss functions
            self.criterion = torch.nn.L1Loss().to(self.device)
            # [SETUP] TODO: Use cross entropy or BCELoss?
            # initialize optimizers
            # [SETUP] TODO: which optimizaer or SDG?
            self.optimizer = torch.optim.Adam(self.netG_S2O.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            # instantiate the method step of the class decaylr
            self.lr_lambda = DecayLR(self.opt.tot_epochs, self.opt.start_from_epoch, self.opt.decay_epochs).step
            # initialise networks scheduler passing the function above
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
            if self.opt.restoring_rep_path is not None:
                file = os.path.join(self.opt.restoring_rep_path,
                                    'checkpoint_epoch_' + str(self.opt.start_from_epoch) + '.pt')
                self.load(file)
        elif self.opt.mode == "eval":
            self.sar_c_vis = self.opt.sar_c
            self.opt_c_vis = self.opt.optical_c
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
                getattr(self, var_name)[temp] = img[i, 0:self.opt_c_vis]

    def backward(self):
        """
        Calculate the loss for generator
        :return:
        """
        self.loss = self.criterion(self.fake_O, self.label)
        self.loss.backward()

    def update_learning_rate(self):
        """
        This function is called to request a step to the scheduler so that to update the learning rate
        :return:
        """
        old_lr = self.optimizer.param_groups[0]['lr']
        self.lr_scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def optimize(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        :return:
        """
        # compute fake images and reconstruction images.
        self.forward('trans')
        # set G_S and G_O's gradients to zero
        self.optimizer.zero_grad()
        # calculate gradients for G_S and G_O
        self.backward()
        # update G's weights
        self.optimizer.step()

    def save_model(self, epoch):
        """
        Save model parameters
        :param epoch: actual epoch
        :return:
        """
        out_file = os.path.join(self.opt.checkpoint_dir, 'checkpoint_epoch_' + str(epoch) + ".pt")
        # save model
        data = {"G_S2O": self.netG_S2O.state_dict(),
                "opt": self.optimizer.state_dict(),
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
            self.optimizer.load_state_dict(data['opt'])
        elif self.opt.mode == 'eval':
            self.netG_S2O.load_state_dict(data['G_S2O'])

    def tb_add_step_loss_g(self, writer, global_step):
        """
        This function add G losses to tensorboard and store the value in the logger

        - loss
        :return: all losses and output of network
        """
        step_loss = {
            'loss': self.loss.item(),
        }
        writer.add_scalars("Train/Generator", step_loss, global_step=global_step)
        self.Logger.append_G(step_loss)

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

    def train(self, train_dataset, eval_dataset, writer=None):
        """
        Run the training for the required epochs
        :param train_dataset: dataset used to train the network
        :param eval_dataset: dataset used to eval the network
        :param writer: a tensorboard instance to track info
        :return:
        """
        self.trans = torch.zeros((len(train_dataset.dataset), self.opt_c_vis, self.opt.patch_size, self.opt.patch_size),
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
                self.optimize()
                # write generator loss to tensorboard
                if global_step > 0 and global_step % self.opt.loss_log_freq == 0:
                    self.tb_add_step_loss_g(writer, global_step)
                    self.Logger.append_loss_step({"step": global_step})
                global_step = global_step + 1

            self.update_learning_rate()
            if epoch >= 0 and epoch % self.opt.save_model_freq == 0:
                self.save_model(epoch)
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
