import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from Lib.Datasets.EUSAR.EUSARDataset import EUSARDataset
from Lib.Nets.SN.SN import SN
from Lib.Nets.utils.config.config_routine import config_routine
from Lib.Nets.utils.config.general_parser import general_parser
from Lib.Nets.utils.config.specific_parser import specific_parser
from Lib.Nets.utils.generic.generic_training import get_subset
from torch.utils.data import DataLoader
import argparse

"""-------------------------------CONFIG----------------------------------"""
parser = argparse.ArgumentParser(description="PyTorch Regression GAN")
parser = general_parser(parser)
opt = specific_parser(
    parser=parser, run_folder='runs/sn/', mode='train', tot_epochs=30, res_block_N=6,
    restoring_rep_path=None, start_from_epoch=None,
    pretrained_GAN=None, GAN_epoch=None, seed=1,
    batch_size_SN=16, prc_train=1, prc_test=1, prc_val=None, sar_c=5, optical_c=4,
    data_dir_train='Data/Train/EUSAR/128_sn_corr', data_dir_test='Data/Test/EUSAR/128_sn_corr',
    acc_log_freq=29, loss_log_freq=1, save_model_freq=100, images_log_freq=None,
    experiment_name='prova',
    run_description='Classifico con nuova metrica accuracy e tutto pt')

opt = config_routine(opt)
"""-------------------------------LOAD DATA----------------------------------"""
train_dataset = EUSARDataset(os.path.join(opt.data_dir_train), True, False, opt.sar_c, opt.optical_c, True)
train_dataset = get_subset(train_dataset, opt.prc_train)
train_dataset = DataLoader(train_dataset, batch_size=opt.batch_size_SN, shuffle=True,
                           num_workers=opt.workers, pin_memory=True, drop_last=False)

test_dataset = EUSARDataset(os.path.join(opt.data_dir_test), True, False, opt.sar_c, opt.optical_c)
test_dataset = get_subset(test_dataset, opt.prc_test, True)
test_dataset = DataLoader(test_dataset, batch_size=opt.batch_size_SN, shuffle=False,
                          num_workers=opt.workers, pin_memory=True, drop_last=False)

"""--------------------------------TRAIN-----------------------------------"""
# Set cuda
device = torch.device("cuda:0")
cudnn.benchmark = True

# Init model
model = SN(opt, device)

# set up tensorboard logging
writer = SummaryWriter(log_dir=os.path.join(opt.tb_dir))

# Model Training
model.train(train_dataset, test_dataset, writer)