import os
from Lib.Datasets.processing.utility import cut_tiles, cut_tiles_radar


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file is a runner to cut processed tiles in patches
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


dataset_name = 'EUSAR'
ps = [128] # [32, 128, 192]
dataset_type = ['Test_corr', 'Train_corr']
for typ in dataset_type:
    for patch_size in ps:
        global_path = '/home/ale/Documents/Python/13_Tesi_2/'
        data_orig = os.path.join(global_path, 'Data/Datasets/', dataset_name, typ)
        dest = os.path.join(global_path, 'Data/', typ.split('_')[0], dataset_name)
        dest_path = os.path.join(dest, str(patch_size) + '_sn_corr/')
        dest_path_trans = os.path.join(dest, str(patch_size) + '_trans_corr/')
        print('Creo:')
        print(' - Global path {}'.format(global_path))
        print(' - Data path {}'.format(data_orig))
        print(' - Dest path {}\n{}'.format(dest_path, dest_path_trans))
        try:
            os.mkdir(dest_path_trans)
            os.mkdir(dest_path)
            os.mkdir(os.path.join(dest_path_trans, 'radar'))
            os.mkdir(os.path.join(dest_path_trans, 'rgb'))
            os.mkdir(os.path.join(dest_path, 'radar'))
            os.mkdir(os.path.join(dest_path, 'label'))
            os.mkdir(os.path.join(dest_path, 'rgb'))
        except:
            print('Already existing folder')

        max_n_bad_pix = 1
        overlapping = 0.5
        padding = True

        cut_tiles(data_orig, dest_path, '1', patch_size, max_n_bad_pix, overlapping, padding)
        cut_tiles_radar(data_orig, dest_path_trans, '1', patch_size, max_n_bad_pix, overlapping, padding)
        print(typ)
        print(dest_path)
        for i in os.listdir(dest_path):
            if '.' not in i:
                print(i)
                print(len(os.listdir(os.path.join(dest_path, i))))

        print(dest_path_trans)
        for i in os.listdir(dest_path_trans):
            if '.' not in i:
                print(i)
                print(len(os.listdir(os.path.join(dest_path_trans, i))))
