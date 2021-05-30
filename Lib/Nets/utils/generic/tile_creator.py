import torch
import os
from Lib.Nets.utils.generic.image2tensorboard import reconstruct_tile
import pickle as pkl

path = '/home/ale/Documents/Python/13_Tesi_2/runs/agan/10_32_idt/checkpoints/args.pkl'
opt = pkl.load(open(path, "rb"))
posx = pkl.load(open(os.path.join(opt.data_dir_train, 'posx.pkl'), "rb"))
posy = pkl.load(open(os.path.join(opt.data_dir_train, 'posy.pkl'), "rb"))

file_list = os.listdir(opt.tb_dir)
tile_list = list(filter(lambda x: '.pt' in x, file_list))
name = 'RT'

par_path = '/home/ale/Documents/Python/13_Tesi_2/Data/Datasets/EUSAR/Train/'
for i in tile_list:
    epoch = i.split('.')[0]
    trans = torch.load(os.path.join(opt.tb_dir, epoch + '.pt'))
    reconstruct_tile(name, opt.patch_size, posx, posy, opt.tb_dir, [8736, 13984], epoch, trans)#, parameter_path=par_path)
