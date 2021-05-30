import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from Lib.Datasets.EUSAR.EUSARDataset import EUSARDataset
from Lib.Nets.Cycle_AT.Cycle_AT import Cycle_AT
from Lib.Nets.utils.config.config_routine import config_routine
from Lib.Nets.utils.config.general_parser import general_parser
from Lib.Nets.utils.config.specific_parser import specific_parser
from Lib.Nets.utils.generic.generic_training import get_subset
import argparse

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This main can be employed to train the Cycle Consistent Adversarial Transcoder.
"CONFIG" section give quick access to some parameter setting.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

"""-------------------------------CONFIG----------------------------------"""
parser = argparse.ArgumentParser(description="PyTorch Cycle GAN")
parser = general_parser(parser)
opt = specific_parser(
    parser=parser, log=False, run_folder='runs/cgan/', mode='train', tot_epochs=200, loss_type='lsgan',
    restoring_rep_path=None, start_from_epoch=None,
    D_training_ratio=1, res_block_N=6, pool_prc_O=0.5, pool_prc_S=0.5,
    buff_dim=1, th_low=0.45, th_high=0.55, pool=False, conditioned=False, dropping=False,
    batch_size=1, prc_train=1, prc_test=1, prc_val=None, sar_c=5, optical_c=4, batch_size_SN=16,
    th_b_h_ratio=200, th_b_l_ratio=2, th_b_h_pool=0.2, th_b_l_pool=0.8, drop_prc=0,
    data_dir_train='Data/Train/EUSAR/128_trans_corr', data_dir_train2='Data/Train/EUSAR/128_sn_corr',
    data_dir_test='Data/Test/EUSAR/128_trans_corr', data_dir_test2='Data/Test/EUSAR/128_sn_corr',
    acc_log_freq=50, loss_log_freq=10, save_model_freq=20, images_log_freq=1,
    experiment_name='std_conv_corr_2_disc',
    run_description='provo la cgfan standard che ha fatto 80 con due disc')
opt = config_routine(opt)
"""-------------------------------LOAD DATA----------------------------------"""
train_dataset = EUSARDataset(os.path.join(opt.data_dir_train), False, True, opt.sar_c, opt.optical_c)
test_dataset = EUSARDataset(os.path.join(opt.data_dir_test), False, True, opt.sar_c, opt.optical_c)
train_dataset = get_subset(train_dataset, opt.prc_train)
test_dataset = get_subset(test_dataset, opt.prc_train)
train_dataset = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                           num_workers=opt.workers, pin_memory=True, drop_last=False)
test_dataset = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.workers, pin_memory=True, drop_last=False)

"""---------------------------------TRAIN------------------------------------"""
# Set cuda
device = torch.device("cuda:0")
cudnn.benchmark = True

# Init model
model = Cycle_AT(opt, device)

# set up tensorboard logging
writer = SummaryWriter(log_dir=os.path.join(opt.global_path, opt.tb_dir))

# Model Training
model.train(train_dataset=train_dataset, eval_dataset=test_dataset, writer=writer)
