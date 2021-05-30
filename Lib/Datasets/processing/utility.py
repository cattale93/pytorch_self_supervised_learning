import numpy as np
import time
import os
import math
import pickle as pkl


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: In this file are implemented some support function to preprocess EUSAR dataset
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def one_hot_2_label_value(label):
    """
    In EUSAR labels are composed of one hot encoded images for each class
    This function converts a structure of [classes, dim_x, dim_y] to a [tile_dim_x, tile_dim_y] in which each
    class has a different value
    Labels are composed as follow (EUSAR):
                 ONE HOT                  SINGLE VALUES          COLOUR
        - Band 0 name forest.png    -->         0         -->     dark green
        - Band 1 name street.png    -->         1         -->     grey
        - Band 2 name field.png     -->         2         -->     lime
        - Band 3 name urban.png     -->         3         -->     red
        - Band 4 name water.png     -->         4         -->     blue
        - All zero values           -->        255        -->     white

    :param label: label patch of shape [C,ps,ps]
    :return: single values image
    """
    label_values = np.zeros(label.shape[1:3])
    for i in range(label.shape[0]):
        # merge each class classified pixel using index + 1
        # index plus 1 because otherwise non classified and index 0 would be equal
        label_values = (label[i] * (i + 1)) + label_values
    # where no classified pixel put 255 else put index
    label_values = np.where(label_values == 0, 255, (label_values - 1))
    label_values.astype(np.uint8)
    return label_values


def remove_duplicates(img):
    """
    This function exist to adjust few label error. Sometimes happens that the same pixels in the dataset are
    associated with two classes.
    The duplicated pixels are removed being judge as non reliable
    :param img: input data
    :return: corrected input
    """
    # sums one hot codes along channel axis
    channel_tot = np.sum(img, 0)
    # if a location is more than 1, the pixel is duplicated, put 0 to duplicated and 1 to non duplicated
    duplicated_location = np.where(channel_tot > 1, 0, 1)
    for i in range(img.shape[0]):
        # duplicated multiplied by 0 other by 1
        img[i] = img[i] * duplicated_location

    return img


def normalizer(img, f, center=False, norm1=False):
    """
    Standardize data channel by channel
    :param img: image to be normalized
    :param f: pointer to the log file
    :param center: if center normalization or not
    :param norm1: normalize between -1 and 1 works only if centering is requested
    :return: [img, mean, std, center_val, mx] along with the normalized image are returned all the normalization
    parameters to be able to recover the original data
    """
    mean = []
    std = []
    mx = []
    f.write("Running normalization\n")
    n_ch = img.shape[0]
    for i in range(0, n_ch):
        mean.append(np.mean(img[i]))
        std.append(np.std(img[i]))
        f.write("Ch {} mean = {} - std = {}\n".format(i, mean[i], std[i]))
        if not norm1:
            img[i] = np.true_divide((img[i] - mean[i]), std[i])

    # if center after the first normalisation to mean 0 and std 1 it scales the value around 0 and between -1 and 1
    if center:
        f.write("Centering values\n")
        center_val = (np.max(img) + np.min(img))/2
        img = img - center_val
        if norm1:
            f.write("Norm values after centering\n")
            for i in range(0, n_ch):
                if norm1:
                    mx.append(np.max(np.abs(img[i])))
                    f.write("Ch {} max = {} \n".format(i, mx[i]))
                    img[i] = np.true_divide(img[i], mx[i])
                else:
                    mx.append(0)
    else:
        center_val = 0

    return img, mean, std, center_val, mx


def simple_normalizer(img, f, mean, std):
    """
    Standardize data channel by channel
    :param img: image to be normalized
    :param f: pointer to the log file
    :param mean: channel mean array
    :param std: channel std array
    :return img: noirmalized img
    """
    f.write("Running normalization\n")
    for i in range(len(mean)):
        f.write("Ch {} mean = {} - std = {}\n".format(i, mean[i], std[i]))
        img[i] = np.true_divide((img[i] - mean[i]), std[i])

    return img


def cut_tiles_full(tile_path, dest_path, name, patch_size, max_n_bad_pix, overlapping, padding):
    """
    This function load data, cut into patches and save them to the destination folder
    Label patches are stored only if there is a certain amount of pixel classified.
    :param tile_path: location of input data tile to be cut
    :param dest_path: folder in which data will be stored
    :param name: target tile name
    :param patch_size: patch dimension
    :param max_n_bad_pix: patch percentage of admissible non classified pixels
    :param overlapping: overlap between two patches
    :param padding: pad data or not
    :param mode: ['eval'| 'train'] in eval mode store data with name which contain the location of that specific patch so that is
    possible to reconstruct the original tile
    :return: NA
    """
    t = time.time()
    f = open(os.path.join(dest_path + name + "_" + "log.txt"), "a")
    f.write("Cutting data from{}\n".format(tile_path + ' ' + name))
    f.write("Patch_size {}\n".format(patch_size))
    f.write("Overlapping {}\n".format(overlapping))
    f.write("patch_max_bad_pix_prc {}\n".format(max_n_bad_pix))
    radar = np.load(os.path.join(tile_path, "radar/" + name + "_radar.npy"))
    label = np.load(os.path.join(tile_path, "label/" + name + "_label.npy"))
    f.write("Loaded radar shape {} type {}\n".format(radar.shape, radar.dtype))
    f.write("Loaded label shape {} type {}\n".format(label.shape, label.dtype))
    rgb = np.load(os.path.join(tile_path, "rgb/" + name + "_rgb.npy"))
    f.write("Loaded rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    w = radar.shape[1]
    h = radar.shape[2]
    if padding:
        # If padding calculate how big the padding has to be
        extra_w = (w % patch_size)
        extra_h = (h % patch_size)
        f.write("Extrapixel w {} h {}\n".format(extra_w, extra_h))
        if extra_w > 0:
            extra_w_l = int(math.ceil((patch_size - extra_w) / 2))
            extra_w_r = int(math.ceil(patch_size - extra_w)) - extra_w_l
        else:
            extra_w_l = 0
            extra_w_r = 0
        if extra_h > 0:
            extra_h_l = int(math.ceil((patch_size - extra_h) / 2))
            extra_h_r = int(math.ceil(patch_size - extra_h)) - extra_h_l
        else:
            extra_h_l = 0
            extra_h_r = 0
        # pad images
        f.write("Extrapixel w_l {} w_r {} h_l {} h_r {}\n".format(extra_w_l, extra_w_r, extra_h_l, extra_h_r))
        radar = np.pad(radar, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        label = np.pad(label, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        f.write("New radar shape {} type {}\n".format(radar.shape, radar.dtype))
        f.write("New label shape {} type {}\n".format(label.shape, label.dtype))
        rgb = np.pad(rgb, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        f.write("New rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    w = radar.shape[1]
    h = radar.shape[2]

    patch_counter = 0
    step = int(patch_size * overlapping)
    # pass all the image with the right stride
    posx, posy = [], []
    for i in range(0, (w - patch_size + step), step):
        for j in range(0, (h - patch_size + step), step):
            validity_patch = label[:, i:i + patch_size, j:j + patch_size]
            u, count = np.unique(validity_patch, return_counts=True)
            if len(u) > 1:
                good_pix_n = count[1]
            else:
                good_pix_n = 0
            patch_counter = patch_counter + 1
            posx.append(str(i))
            posy.append(str(j))
            t_n, p_n = patch_name(name, patch_counter)
            # label are saved only with the right number of classified pixel
            if good_pix_n >= max_n_bad_pix:
                np.save(os.path.join(dest_path, "label", t_n + "_" + p_n + "_label.npy"),
                        label[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
            # rgb and radar patches are always stored
            np.save(os.path.join(dest_path, "radar", t_n + "_" + p_n + "_radar.npy"),
                    radar[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
            np.save(os.path.join(dest_path, "rgb", t_n + "_" + p_n + "_rgb.npy"),
                    rgb[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
            f.write("{} Patch ({},{} - {},{}) good_pix_n = {:.3f}\n".format(
                name + "_" + str(patch_counter), i, i+patch_size, j, j+patch_size, good_pix_n))
    f.write("Execution time = {:.2f} s".format(time.time() - t))
    pkl.dump(posx, open(os.path.join(dest_path, "posx.pkl"), "wb"))
    pkl.dump(posy, open(os.path.join(dest_path, "posy.pkl"), "wb"))
    print("Execution time = {:.2f} s".format(time.time() - t))
    f.close()


def cut_tiles_small(tile_path, dest_path, name, patch_size, max_n_bad_pix, overlapping, padding):
    """
    This function load data cut into patches and save them to the destination folder
    Label patches are stored only if there is a certain amount of pixel classified. So is suitable if all data are used
    Small because works only on a piece of the image
    :param tile_path: location of input data tile to be cut
    :param dest_path: folder in which data will be stored
    :param name: target tile name
    :param patch_size: patch dimension
    :param max_n_bad_pix: patch percentage of admissible non classified pixels
    :param overlapping: overlap between two patches
    :param padding: pad data or not
    :param mode: ['eval'| 'train'] in eval mode store data with name which contain the location of that specific patch so that is
    possible to reconstruct the original tile
    :return: NA
    """
    t = time.time()
    f = open(os.path.join(dest_path + name + "_" + "log.txt"), "a")
    f.write("Cutting data from{}\n".format(tile_path + ' ' + name))
    f.write("Patch_size {}\n".format(patch_size))
    f.write("Overlapping {}\n".format(overlapping))
    f.write("patch_max_bad_pix_prc {}\n".format(max_n_bad_pix))
    radar = np.load(os.path.join(tile_path, "radar/" + name + "_radar.npy"))
    label = np.load(os.path.join(tile_path, "label/" + name + "_label.npy"))
    f.write("Loaded radar shape {} type {}\n".format(radar.shape, radar.dtype))
    f.write("Loaded label shape {} type {}\n".format(label.shape, label.dtype))
    rgb = np.load(os.path.join(tile_path, "rgb/" + name + "_rgb.npy"))
    f.write("Loaded rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    w = radar.shape[1]
    h = radar.shape[2]
    if padding:
        # If padding calculate how big the padding has to be
        extra_w = (w % patch_size)
        extra_h = (h % patch_size)
        f.write("Extrapixel w {} h {}\n".format(extra_w, extra_h))
        if extra_w > 0:
            extra_w_l = int(math.ceil((patch_size - extra_w) / 2))
            extra_w_r = int(math.ceil(patch_size - extra_w)) - extra_w_l
        else:
            extra_w_l = 0
            extra_w_r = 0
        if extra_h > 0:
            extra_h_l = int(math.ceil((patch_size - extra_h) / 2))
            extra_h_r = int(math.ceil(patch_size - extra_h)) - extra_h_l
        else:
            extra_h_l = 0
            extra_h_r = 0
        # pad images
        f.write("Extrapixel w_l {} w_r {} h_l {} h_r {}\n".format(extra_w_l, extra_w_r, extra_h_l, extra_h_r))
        radar = np.pad(radar, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        label = np.pad(label, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        f.write("New radar shape {} type {}\n".format(radar.shape, radar.dtype))
        f.write("New label shape {} type {}\n".format(label.shape, label.dtype))
        rgb = np.pad(rgb, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        f.write("New rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    patch_counter = 0
    step = int(patch_size * overlapping)
    # pass all the image with the right stride
    posx, posy = [], []
    for i in range(850, (1490 - patch_size + step), step):
        for j in range(850, (1490 - patch_size + step), step):

            validity_patch = label[:, i:i + patch_size, j:j + patch_size]
            u, count = np.unique(validity_patch, return_counts=True)
            if len(u) > 1:
                good_pix_n = count[1]
            else:
                good_pix_n = 0
            patch_counter = patch_counter + 1
            posx.append(str(i))
            posy.append(str(j))
            t_n, p_n = patch_name(name, patch_counter)
            # label are saved only with the right number of classified pixel
            if good_pix_n >= max_n_bad_pix:
                np.save(os.path.join(dest_path, "label", t_n + "_" + p_n + "_label.npy"),
                        label[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
            # rgb and radar patches are always stored
            np.save(os.path.join(dest_path, "radar", t_n + "_" + p_n + "_radar.npy"),
                    radar[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
            np.save(os.path.join(dest_path, "rgb", t_n + "_" + p_n + "_rgb.npy"),
                    rgb[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
            f.write("{} Patch ({},{} - {},{}) good_pix_n = {:.3f}\n".format(
                name + "_" + str(patch_counter), i, i+patch_size, j, j+patch_size, good_pix_n))
    f.write("Execution time = {:.2f} s".format(time.time() - t))
    pkl.dump(posx, open(os.path.join(dest_path, "posx.pkl"), "wb"))
    pkl.dump(posy, open(os.path.join(dest_path, "posy.pkl"), "wb"))
    print("Execution time = {:.2f} s".format(time.time() - t))
    f.close()


def cut_tiles(tile_path, dest_path, name, patch_size, max_n_bad_pix, overlapping, padding):
    """
    This function load data cut into patches and save them to the destination folder
    Patches are stored only if there is a certain amount of pixel classified. So is suitable if all data are used
    :param tile_path: location of input data tile to be cut
    :param dest_path: folder in which data will be stored
    :param name: target tile name
    :param patch_size: patch dimension
    :param max_n_bad_pix: patch percentage of admissible non classified pixels
    :param overlapping: overlap between two patches
    :param padding: pad data or not
    :return: NA
    """
    t = time.time()
    f = open(os.path.join(dest_path + name + "_" + "log.txt"), "a")
    f.write("Cutting data from{}\n".format(tile_path + ' ' + name))
    f.write("Patch_size {}\n".format(patch_size))
    f.write("Overlapping {}\n".format(overlapping))
    f.write("patch_max_bad_pix_prc {}\n".format(max_n_bad_pix))
    radar = np.load(os.path.join(tile_path, "radar/" + name + "_radar.npy"))
    label = np.load(os.path.join(tile_path, "label/" + name + "_label.npy"))
    f.write("Loaded radar shape {} type {}\n".format(radar.shape, radar.dtype))
    f.write("Loaded label shape {} type {}\n".format(label.shape, label.dtype))
    rgb = np.load(os.path.join(tile_path, "rgb/" + name + "_rgb.npy"))
    f.write("Loaded rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    w = radar.shape[1]
    h = radar.shape[2]
    if padding:
        # If padding calculate how big the padding has to be
        extra_w = (w % patch_size)
        extra_h = (h % patch_size)
        f.write("Extrapixel w {} h {}\n".format(extra_w, extra_h))
        if extra_w > 0:
            extra_w_l = int(math.ceil((patch_size - extra_w) / 2))
            extra_w_r = int(math.ceil(patch_size - extra_w)) - extra_w_l
        else:
            extra_w_l = 0
            extra_w_r = 0
        if extra_h > 0:
            extra_h_l = int(math.ceil((patch_size - extra_h) / 2))
            extra_h_r = int(math.ceil(patch_size - extra_h)) - extra_h_l
        else:
            extra_h_l = 0
            extra_h_r = 0
        # pad images
        f.write("Extrapixel w_l {} w_r {} h_l {} h_r {}\n".format(extra_w_l, extra_w_r, extra_h_l, extra_h_r))
        radar = np.pad(radar, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        label = np.pad(label, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        f.write("New radar shape {} type {}\n".format(radar.shape, radar.dtype))
        f.write("New label shape {} type {}\n".format(label.shape, label.dtype))
        rgb = np.pad(rgb, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        f.write("New rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    w = radar.shape[1]
    h = radar.shape[2]
    patch_counter = 0
    step = int(patch_size * overlapping)
    # pass all the image with the right stride
    # 0, (h     0, (w
    posx, posy = [], []
    for i in range(0, (w - patch_size + step), step):
        for j in range(0, (h - patch_size + step), step):
            validity_patch = label[:, i:i + patch_size, j:j + patch_size]
            u, count = np.unique(validity_patch, return_counts=True)
            if len(u) > 1:
                good_pix_n = count[1]
            else:
                good_pix_n = 0
            # store data only if the number of classified pixel of the actual patch are enough
            if good_pix_n >= max_n_bad_pix:
                patch_counter = patch_counter + 1
                posx.append(str(i))
                posy.append(str(j))
                t_n, p_n = patch_name(name, patch_counter)
                np.save(os.path.join(dest_path, "radar", t_n + "_" + p_n + "_radar.npy"),
                        radar[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
                np.save(os.path.join(dest_path, "label", t_n + "_" + p_n + "_label.npy"),
                        label[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
                np.save(os.path.join(dest_path, "rgb", t_n + "_" + p_n + "_rgb.npy"),
                        rgb[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
                f.write("{} Patch ({},{} - {},{}) good_pix_n = {:.3f}\n".format(
                    name + "_" + str(patch_counter), i, i+patch_size, j, j+patch_size, good_pix_n))
    f.write("Execution time = {:.2f} s".format(time.time() - t))
    pkl.dump(posx, open(os.path.join(dest_path, "posx.pkl"), "wb"))
    pkl.dump(posy, open(os.path.join(dest_path, "posy.pkl"), "wb"))
    print("Execution time = {:.2f} s".format(time.time() - t))
    f.close()


def patch_name(tile_id, patch_id):
    """
    convert the nth tile in a string which is padded with 0 up to tile_max_len positions
    convert the nth tile in a string which is padded with 0 up to patch_max_len positions
    :param tile_id: number of actual tile
    :param patch_id: number of actual patch
    :return: two new strings
    """
    # figures of tiles
    tile_max_len = 4
    # figures of patchs
    patch_max_len = 7
    tile_id_str = str(tile_id)
    patch_id_str = str(patch_id)
    tile_id_len = len(tile_id_str)
    patch_id_len = len(patch_id_str)
    tile_id_str = '0' * (tile_max_len-tile_id_len) + tile_id_str
    patch_id_str = '0' * (patch_max_len-patch_id_len) + patch_id_str
    return tile_id_str, patch_id_str


def cut_tiles_radar(tile_path, dest_path, name, patch_size, max_n_bad_pix, overlapping, padding):
    """
    This function load data cut into patches and save them to the destination folder
    Label patches are stored only if there is a certain amount of pixel classified. So is suitable if all data are used
    Full because saves almost everything except some label
    :param tile_path: location of input data tile to be cut
    :param dest_path: folder in which data will be stored
    :param name: target tile name
    :param patch_size: patch dimension
    :param max_n_bad_pix: patch percentage of admissible non classified pixels
    :param overlapping: overlap between two patches
    :param padding:
    possible to reconstruct the original tile
    :return: NA
    """
    t = time.time()
    f = open(os.path.join(dest_path + name + "_" + "log.txt"), "a")
    f.write("Cutting data from{}\n".format(tile_path + ' ' + name))
    f.write("Patch_size {}\n".format(patch_size))
    f.write("Overlapping {}\n".format(overlapping))
    f.write("patch_max_bad_pix_prc {}\n".format(max_n_bad_pix))
    radar = np.load(os.path.join(tile_path, "radar/" + name + "_radar.npy"))
    label = np.load(os.path.join(tile_path, "label/" + name + "_label.npy"))
    rgb = np.load(os.path.join(tile_path, "rgb/" + name + "_rgb.npy"))
    f.write("Loaded radar shape {} type {}\n".format(radar.shape, radar.dtype))
    f.write("Loaded label shape {} type {}\n".format(label.shape, label.dtype))
    f.write("Loaded rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    img_r = radar[[2, 1, 0], :, :]
    img_r = np.rollaxis(img_r, 0, 3)
    minimum = np.min(img_r)
    if minimum < 0:
        img_r = img_r - minimum
    maximum = np.max(img_r)
    img_r = np.divide(img_r, maximum)
    img_r = img_r * 255
    img_r = img_r.astype(np.uint8)

    img_o = rgb[[2, 1, 0], :, :]
    img_o = np.rollaxis(img_o, 0, 3)
    minimum = np.min(img_o)
    if minimum < 0:
        img_o = img_o - minimum
    maximum = np.max(img_o)
    img_o = np.divide(img_o, maximum)
    img_o = img_o * 255
    img_o = img_o.astype(np.uint8)

    white = np.ones((patch_size, patch_size, 3)) * 255
    white = white.astype(np.uint8)

    w = radar.shape[1]
    h = radar.shape[2]
    if padding:
        # If padding calculate how big the padding has to be
        extra_w = (w % patch_size)
        extra_h = (h % patch_size)
        f.write("Extrapixel w {} h {}\n".format(extra_w, extra_h))
        if extra_w > 0:
            extra_w_l = int(math.ceil((patch_size - extra_w) / 2))
            extra_w_r = int(math.ceil(patch_size - extra_w)) - extra_w_l
        else:
            extra_w_l = 0
            extra_w_r = 0
        if extra_h > 0:
            extra_h_l = int(math.ceil((patch_size - extra_h) / 2))
            extra_h_r = int(math.ceil(patch_size - extra_h)) - extra_h_l
        else:
            extra_h_l = 0
            extra_h_r = 0
        # pad images
        f.write("Extrapixel w_l {} w_r {} h_l {} h_r {}\n".format(extra_w_l, extra_w_r, extra_h_l, extra_h_r))
        radar = np.pad(radar, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        label = np.pad(label, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        # img_o = np.pad(img_o, ((extra_w_l, extra_w_r), (extra_h_l, extra_h_r), (0, 0)), mode="wrap")
        # img_r = np.pad(img_r, ((extra_w_l, extra_w_r), (extra_h_l, extra_h_r), (0, 0)), mode="wrap")
        f.write("New radar shape {} type {}\n".format(radar.shape, radar.dtype))
        f.write("New label shape {} type {}\n".format(label.shape, label.dtype))
        rgb = np.pad(rgb, ((0, 0), (extra_w_l, extra_w_r), (extra_h_l, extra_h_r)), mode="wrap")
        f.write("New rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    f.write("New radar shape {} type {}\n".format(radar.shape, radar.dtype))
    f.write("New label shape {} type {}\n".format(label.shape, label.dtype))
    f.write("New rgb shape {} type {}\n".format(rgb.shape, rgb.dtype))

    w = radar.shape[1]
    h = radar.shape[2]

    patch_counter = 0
    step = int(patch_size * overlapping)
    # pass all the image with the right stride
    posx, posy = [], []
    for i in range(0, (w - patch_size + step), step):
        for j in range(0, (h - patch_size + step), step):
            validity_patch = label[:, i:i + patch_size, j:j + patch_size]
            u, count = np.unique(validity_patch, return_counts=True)
            if len(u) > 1:
                good_pix_n = count[1]
            else:
                good_pix_n = 0
            # label are saved only with the right number of classified pixel
            # rgb and radar patches are always stored
            if not (1500 < i < 7000 and 1500 < j < 12500):
                r = radar[:, i:i + patch_size, j:j + patch_size]
                o = rgb[:, i:i + patch_size, j:j + patch_size]
                r = np.sum(r, 0)
                o = np.sum(o, 0)
                r = np.unique(r)
                o = np.unique(o)
                if len(r) > (patch_size * 28) and len(o) > (patch_size * 4):
                    posx.append(str(i))
                    posy.append(str(j))
                    patch_counter = patch_counter + 1
                    t_n, p_n = patch_name(name, patch_counter)
                    np.save(os.path.join(dest_path, "radar", t_n + "_" + p_n + "_radar.npy"),
                            radar[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
                    np.save(os.path.join(dest_path, "rgb", t_n + "_" + p_n + "_rgb.npy"),
                            rgb[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
                # else:
                # img_o[i:i + patch_size, j:j + patch_size, :] = white
                # img_r[i:i + patch_size, j:j + patch_size, :] = white
            else:
                posx.append(str(i))
                posy.append(str(j))
                patch_counter = patch_counter + 1
                t_n, p_n = patch_name(name, patch_counter)
                np.save(os.path.join(dest_path, "radar", t_n + "_" + p_n + "_radar.npy"),
                        radar[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)
                np.save(os.path.join(dest_path, "rgb", t_n + "_" + p_n + "_rgb.npy"),
                        rgb[:, i:i + patch_size, j:j + patch_size], allow_pickle=True)

            f.write("{} Patch ({},{} - {},{}) good_pix_n = {}\n".format(
                name + "_" + str(patch_counter), i, i + patch_size, j, j + patch_size, good_pix_n))

    # temp = Image.fromarray(img_r, 'RGB')
    # temp.save('radar' + '.png')
    # temp = Image.fromarray(img_o, 'RGB')
    # temp.save('rgb' + '.png')
    f.write("Execution time = {:.2f} s".format(time.time() - t))
    pkl.dump(posx, open(os.path.join(dest_path, "posx.pkl"), "wb"))
    pkl.dump(posy, open(os.path.join(dest_path, "posy.pkl"), "wb"))
    print("Execution time = {:.2f} s".format(time.time() - t))
    f.close()
