import torch
from torch import nn
import os
from Lib.Nets.utils.arch.arch import Generator, newSN
from Lib.Nets.utils.generic.image2tensorboard import display_label_single_c, display_input, display_predictions
from tqdm import tqdm
from Lib.utils.generic.generic_utils import start, stop
from Lib.utils.metrics.Accuracy import Accuracy
from Lib.utils.Logger.Logger import Logger
from Lib.Nets.utils.generic.generic_training import set_requires_grad, calculate_accuracy, breaker
import pickle as pkl


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file implements the classifier network to put on top of the feature extractors 
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class SN:
    """
    This class implements all the variables, functions, methods, required to deploy a shallow U-Net which is used a classifier
    on top of a pretrained model
    """
    def __init__(self, opt, device):
        """
        define network
        :param opt: opt contain all the variables that are configurable when launching the script, check the
        folder: Lib/Nets/utils/config/ there are three scripts which are used to configure networks
        :param device: cuda device
        :return:
        """
        # images placeholders
        self.real_S = None
        self.label = None
        self.seg_map = None
        # cost function values
        self.loss = None
        # general
        self.device = device
        self.opt = opt
        self.Accuracy_test = Accuracy()
        self.Accuracy_train = Accuracy()
        self.Logger_test = Logger(self.opt.mode)
        self.Logger_train = Logger(self.opt.mode)
        self.sar_c_vis = min(self.opt.sar_c, 3)
        self.posx_train = pkl.load(open(os.path.join(opt.data_dir_train, 'posx.pkl'), "rb"))
        self.posy_train = pkl.load(open(os.path.join(opt.data_dir_train, 'posy.pkl'), "rb"))
        self.posx_test = pkl.load(open(os.path.join(opt.data_dir_test, 'posx.pkl'), "rb"))
        self.posy_test = pkl.load(open(os.path.join(opt.data_dir_test, 'posy.pkl'), "rb"))
        # net
        self.netG_S2O = Generator(self.opt.sar_c, self.opt.optical_c, self.opt.dropout, self.opt.bias)
        #self.SN = SN(self.opt.sar_c, self.opt.N_classes).to(self.device)
        self.SegNet = newSN(self.opt.N_classes, self.opt.bias, self.opt.dropout).to(self.device)

        if self.opt.mode == "train":
            print('Mode -> train')
            if self.opt.GAN_epoch is not None:
                file = os.path.join(self.opt.global_path, self.opt.pretrained_GAN,
                                    'checkpoint_epoch_' + str(self.opt.GAN_epoch) + '.pt')
                self.load_GAN(file)
                print('Loaded model {}'.format(file))
                #self.load_SN(file)
                #self.SN.to(self.device)
            else:
                # [SETUP] TODO: init weight of segnet
                # init weights
                pass
            set_requires_grad(self.SegNet, True)
            set_requires_grad(self.netG_S2O, False)

            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').to(self.device)
            # self.optimizer = torch.optim.RMSprop(self.SN.parameters(), lr=self.opt.lr_SN,
            #                                     weight_decay=self.opt.weight_decay_SN)
            self.optimizer = torch.optim.Adam(self.SegNet.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        elif self.opt.mode == "eval":
            print('Mode -> train')
            set_requires_grad(self.netG_S2O, False)
            set_requires_grad(self.SegNet, False)
            file_GAN = os.path.join(self.opt.global_path, self.opt.pretrained_GAN,
                                    'checkpoint_epoch_' + str(self.opt.GAN_epoch) + '.pt')
            file_SN = os.path.join(self.opt.global_path, self.opt.restoring_rep_path,
                                   'checkpoint_epoch_' + str(self.opt.start_from_epoch) + '.pt')
            self.load_all(file_GAN, file_SN)
        temp = 22 - (9 - self.opt.res_block_N)
        self.netG_S2O = self.netG_S2O.model[0:temp]
        self.netG_S2O.to(self.device)

    def set_input(self, data):
        """
        Unpack input data from the dataloader
        :param data:
        :return:
        """
        self.real_S = data['radar'].to(self.device)
        self.label = data['label'].to(self.device).type(torch.long)

    def forward(self):
        """
        Run forward pass
        :return:
        """
        # use the output of the first up sampling layer (second last relu)
        # detach should detach from Generator.
        feature_map = self.netG_S2O(self.real_S).detach()
        self.seg_map = self.SegNet(feature_map)

    def backward(self):
        """
        Calculates loss and calculate gradients running loss.backward()
        :return: NA
        """
        self.loss = self.criterion(self.seg_map, self.label)

        self.loss.backward()

    def optimize(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        :return:
        """
        # compute fake images and reconstruction images.
        self.forward()
        set_requires_grad(self.netG_S2O, False)
        # set gradients to zero
        self.optimizer.zero_grad()
        # calculate gradients
        self.backward()
        # update only segnet weights weights
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
                "SN": self.SegNet.state_dict(),
                "opt_SegNet": self.optimizer.state_dict(),
                }

        torch.save(data, out_file)

    def load_GAN(self, file):
        """
        Restore generator parameters
        :param file: file from where load parameters
        :return:
        """
        data = torch.load(file)
        self.netG_S2O.load_state_dict(data['G_S2O'])

    def load_SN(self, file):
        """
        Restore generator parameters
        :param file: file from where load parameters
        :return:
        """
        pt_dict = torch.load(file)
        pretrained_dict = {k: v for k, v in pt_dict['G_S2O'].items() if '19' in k or '22' in k}

        SN_dict = self.SegNet.state_dict()

        SN_dict_names = []
        for k, v in SN_dict.items():
            SN_dict_names.append(k)

        i = 0
        for k, v in pretrained_dict.items():
            SN_dict[SN_dict_names[i]] = v
            i = i + 1

        self.SegNet.load_state_dict(SN_dict)

    def load_all(self, file_GAN, file_SN):
        """
        Restore generator and classifier parameters
        :param file_GAN: file from where load GAN parameters
        :param file_SN: file from where load SN parameters
        :return:
        """
        data_GAN = torch.load(file_GAN)
        data_SN = torch.load(file_SN)
        if self.opt.mode == 'train':
            self.netG_S2O.load_state_dict(data_GAN['G_S2O'])
            self.SegNet.load_state_dict(data_SN['SN'])
            self.optimizer.load_state_dict(data_SN['opt_SegNet'])
        elif self.opt.mode == 'eval':
            self.netG_S2O.load_state_dict(data_GAN['G_S2O'])
            self.SegNet.load_state_dict(data_SN['SN'])

    def tb_add_step_loss(self, writer, global_step):
        """
        Saves segnet loss to tensorboard and segnet acc
        :param writer: pointer to tb
        :param global_step: step for tb
        :return:
        """
        # log loss
        temp_loss = self.loss.item()
        writer.add_scalar("Train/Loss", temp_loss, global_step=global_step)
        loss_SN = {"loss_SN": temp_loss}
        self.Logger_train.append_SN_loss(loss_SN)

    def tb_add_step_images(self, writer=None, global_step=None):
        """
        Saves all net images to tensorboard
        - real_S
        - label
        - prediction
        :param writer: pointer to tb writer
        :param global_step: step for tb
        :return:
        """
        label_norm, mask = display_label_single_c(self.label)
        seg_map_norm = display_predictions(self.seg_map, True, mask)
        real_S_norm = display_input(self.real_S[:, 0:self.sar_c_vis, :, :], False)
        # if the writer is not passed to the function instead of updating tb it returns the images,
        # this variant is useful to create tile
        if writer is None:
            return label_norm[0], seg_map_norm[0]
        else:
            writer.add_images("Train/1 - Labes", label_norm, global_step=global_step)
            writer.add_images("Train/2 - Map", seg_map_norm, global_step=global_step)
            writer.add_images("Train/3 - Radar Input", real_S_norm, global_step=global_step)

    def train(self, train_dataset, eval_dataset, writer):
        """
        Run the training for the required epochs
        :param train_dataset: dataset used to train the network
        :param eval_dataset:
        :param writer: a tensorboard instance to track info
        :return:
        """
        global_step = 0
        if self.opt.acc_log_freq == 117:
            calculate_accuracy(self, eval_dataset, writer, global_step, "Test", 0, self.posx_test, self.posy_test, self.opt.test_size)
            calculate_accuracy(self, train_dataset, writer, global_step, "Train", 0, self.posx_train, self.posy_train, self.opt.train_size)
            self.Logger_train.append_acc_step({"step": global_step})
            self.Logger_test.append_acc_step({"step": global_step})
        for epoch in range(self.opt.tot_epochs):
            t = start()
            text_line = "=" * 27 + "SN EPOCH " + str(epoch) + "/" + str(self.opt.tot_epochs) + "=" * 27
            print(text_line)
            progress_bar = tqdm(enumerate(train_dataset), total=len(train_dataset))
            # Train for each patch in the
            for i, data in progress_bar:
                self.set_input(data)
                self.optimize()
                self.tb_add_step_loss(writer, global_step)
                #self.Logger_train.append_loss_step({"step": global_step})
                global_step = global_step + 1

            if (epoch > 0 and epoch % self.opt.acc_log_freq == 0) or self.opt.acc_log_freq == 117:
                calculate_accuracy(self, eval_dataset, writer, global_step, "Test", epoch, self.posx_test, self.posy_test, self.opt.test_size)
                #calculate_accuracy(self, train_dataset, writer, global_step, "Train", epoch, self.posx_train, self.posy_train, self.opt.train_size)
                #self.Logger_train.append_acc_step({"step": global_step})
                self.Logger_test.append_acc_step({"step": global_step})

            if epoch > 0 and epoch % self.opt.save_model_freq == 0:
                self.save_model(epoch)
            self.Logger_test.save_logger(self.opt.checkpoint_dir, name='test')
            #self.Logger_train.save_logger(self.opt.checkpoint_dir, name='train')
            # Epoch duration
            s = 'SN Epoch {} '.format(epoch)
            stop(t, s)
            if breaker(self.opt, epoch):
                print('EXECUTION FORCED TO STOP AT {} EPOCHS'.format(epoch))
                break
