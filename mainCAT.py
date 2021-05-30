import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from Lib.Datasets.EUSAR.EUSARDataset import EUSARDataset
from Lib.Nets.CAT.CAT import CAT
from Lib.Nets.utils.config.config_routine import config_routine
from Lib.Nets.utils.config.general_parser import general_parser
from Lib.Nets.utils.config.specific_parser import specific_parser
from Lib.Nets.utils.generic.generic_training import get_subset
from torch.utils.data.dataloader import DataLoader
import argparse

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This main is employed to train the Conditional Adversarial Transcoder.
"CONFIG" section give quick access to some parameter setting.
 """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""-------------------------------CONFIG----------------------------------"""
parser = argparse.ArgumentParser(description="CONDITIONAL ADVERSARIAL TRANSCODER")
parser = general_parser(parser)
opt = specific_parser(
    parser=parser, log=False, run_folder='runs/agan/', mode='train', tot_epochs=211, loss_type='lsgan',
    restoring_rep_path=None, start_from_epoch=None, D_training_ratio=5, lambda_A=10, lambda_gp=None,
    buff_dim=1, th_low=None, th_high=None, pool=False, dropping=False,  pool_prc_O=0.5,
    batch_size=1, prc_train=1, prc_test=1, prc_val=None, sar_c=5, optical_c=4, batch_size_SN=16, res_block_N=6,
    data_dir_train='Data/Train/EUSAR/128_trans_corr', data_dir_train2='Data/Train/EUSAR/128_sn_corr',
    data_dir_test='Data/Test/EUSAR/128_trans_corr', data_dir_test2='Data/Test/EUSAR/128_sn_corr',
    acc_log_freq=50, loss_log_freq=10, save_model_freq=20, images_log_freq=20,
    experiment_name='128_corr_data_2_disc',
    run_description='provo cgan std con 2 disc')
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
model = CAT(opt, device)

# set up tensorboard logging
writer = SummaryWriter(log_dir=os.path.join(opt.global_path, opt.tb_dir))

# Model Training
model.train(train_dataset=train_dataset, eval_dataset=test_dataset, writer=writer)
