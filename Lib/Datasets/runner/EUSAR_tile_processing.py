import os
from Lib.Datasets.processing.EUSAR_data_processing import png_to_numpy, process_tile


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file is a runner to preprocess raw tiles
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


orig = ['orig', 'orig_test']
dest = ['Train_corr', 'Test_corr']
for i in range(2):
    print(i)
    global_path = '/home/ale/Documents/Python/13_Tesi_2/'
    orig_path = 'Data/Datasets/EUSAR/' + orig[i] + '/'
    print(orig_path)
    dest_path = os.path.join(global_path, 'Data/Datasets/EUSAR/', dest[i])
    print(dest_path)
    try:
        os.mkdir(dest_path)
        os.mkdir(os.path.join(dest_path, 'radar'))
        os.mkdir(os.path.join(dest_path, 'label'))
        os.mkdir(os.path.join(dest_path, 'rgb'))
    except:
        print('Already existing folder')

    label_path = os.path.join(global_path, orig_path, 'label/')
    # create unique numpy array of labels
    png_to_numpy(label_path, dest_path, '1')

    tile_path = os.path.join(global_path, orig_path, 'radar')
    tile_name = 'radar.tif'
    dest_folder_radar = os.path.join(global_path, dest_path, 'radar/')
    # process SAR tile
    process_tile(tile_path, tile_name, dest_folder_radar, 'SAR_feature', '1', 'box', 3)

    tile_path = os.path.join(global_path, orig_path, 'rgb')
    tile_name = 'rgb.tif'
    dest_folder_rgb = os.path.join(global_path, dest_path, 'rgb/')
    # process rgb tile
    process_tile(tile_path, tile_name, dest_folder_rgb, 'tif2npy', '1')
