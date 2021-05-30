import os
from torch.utils.tensorboard import SummaryWriter
from Lib.Nets.SN.SN import SN
from Lib.Nets.utils.config.config_routine import config_routine
from Lib.Nets.utils.config.general_parser import general_parser
from Lib.Nets.utils.config.specific_parser import specific_parser
from Lib.Datasets.EUSAR.EUSARDataset import EUSARDataset
import argparse
from torch.utils.data import DataLoader
from Lib.Nets.utils.generic.generic_training import get_subset


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This function is employed to test feature extraction capability. In fact can be called by network
implementations to train a classifier on the top of the capacity just leaned.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def trainSN(options, epoch, device):
    """
    Run a quick SN training
    :param options: pretrained model options
    :param epoch: epoch to be loaded
    :param device:
    :return:
    """
    """-------------------------------CONFIG----------------------------------"""
    parser = argparse.ArgumentParser(description="PyTorch Regression GAN")
    parser = general_parser(parser)
    opt = specific_parser(
        parser=parser, run_folder=options.log_dir, mode='train', tot_epochs=30, pretrained_GAN=options.checkpoint_dir,
        GAN_epoch=epoch, acc_log_freq=options.acc_log_freq, loss_log_freq=options.loss_log_freq,
        batch_size_SN=options.batch_size_SN, images_log_freq=options.images_log_freq,
        data_dir_train=options.data_dir_train2, data_dir_test=options.data_dir_test2,
        experiment_name='SN'+str(epoch), sar_c=options.sar_c, optical_c=options.optical_c,
        save_model_freq=1000, res_block_N=options.res_block_N)

    opt = config_routine(opt)

    """-----------------------------DATA LOADER--------------------------------"""
    train_dataset = EUSARDataset(os.path.join(options.data_dir_train2), True, False, options.sar_c, options.optical_c)
    train_dataset = get_subset(train_dataset, options.prc_test)
    train_dataset = DataLoader(train_dataset, batch_size=options.batch_size_SN, shuffle=True,
                               num_workers=options.workers, pin_memory=True, drop_last=False)

    test_dataset = EUSARDataset(os.path.join(options.data_dir_test2), True, False, options.sar_c, options.optical_c)
    test_dataset = get_subset(test_dataset, options.prc_test, True)
    test_dataset = DataLoader(test_dataset, batch_size=options.batch_size_SN, shuffle=False,
                              num_workers=options.workers, pin_memory=True, drop_last=False)

    """--------------------------------TRAIN-----------------------------------"""
    # Init model
    model = SN(opt, device)

    # set up tensorboard logging
    writer = SummaryWriter(log_dir=os.path.join(opt.tb_dir))
    # Model Training
    model.train(train_dataset, test_dataset, writer)
