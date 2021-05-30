import os
import time
import numpy as np
from PIL import Image
from skimage import io
from scipy import ndimage
from Lib.Datasets.processing.utility import remove_duplicates, simple_normalizer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file implements function to process EUSAR data preprocessing such as radar feature extraction
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def png_to_numpy(data_path, dest_path, name):
    """
    This function transform many black and white png images of label in a unique npy array with [Classes, W,H]
    The resulting image one hot encoded
    :param data_path: where data are now stored
    :param dest_path: where new data wil be stored
    :param name: name is the output name of the label variable global_path/(name + _label.npy)
    :return: NA
    """
    t = time.time()
    image_list = os.listdir(data_path)
    f = open(os.path.join(dest_path, "label", name + "_label.txt"), "a")
    f.write("Images to be merged {}\n".format(image_list))
    w, h = Image.open(os.path.join(data_path, image_list[0])).size
    f.write("Sample size w {}, h {}\n".format(h, w))
    image = np.zeros((len(image_list), h, w), dtype=np.uint8)
    f.write("Created numpy array of format {} and type {}\n".format(image.shape, image.dtype))

    for i, img_name in enumerate(image_list):
        f.write("Band {} name {}\n".format(i, img_name))
        img = Image.open(os.path.join(data_path, img_name)).convert("L")
        img = np.asarray(img)
        image[i] = img

    image = (image / 255).astype(np.float32)
    image = remove_duplicates(image)
    f.write("Shape {} type {}\n".format(image.shape, image.dtype))
    u, c = np.unique(image, return_counts=True)
    f.write("Unique {} count {}\n".format(u, c))
    image = image.astype(np.uint8)
    np.save(os.path.join(dest_path, "label", name + "_label.npy"), image, allow_pickle=True)

    f.write("Execution time = {:.2f} s".format(time.time() - t))
    f.close()


def process_tile(tile_path, tile_name, dest_path, action, name, filter_type='box', filter_kernel=3, center=False):
    """
    Open tif or numpy tile and apply an action which is a function which process the data and store them in the new
    location output dest
    :param tile_path: position of the desired path
    :param tile_name: name of the tile
    :param dest_path: where to store the new tile
    :param action: what kind of processing apply {"SAR_feature", "tif2npy"}
    :param name: name of the tile
    :param filter_type: ['box', 'gaus']
    :param filter_kernel: ['3', '0.4'] filter size 3x3 for more detail see specific fucntions
    :param center: center values or not (now if center it re-normalize between -1 and 1)
    :return: NA
    """

    t = time.time()
    f = open(os.path.join(dest_path, name + "_" + tile_name.split(".")[0] + ".txt"), "a")
    formato = tile_name.split(".")[-1]
    if formato == "tif":
        f.write("Image format {}\n".format(formato))
        tile = io.imread(os.path.join(tile_path, tile_name))
        f.write("Input tile shape {} type {}\n".format(tile.shape, tile.dtype))
        new_tile = np.asarray(tile, order="F")
        new_tile = np.rollaxis(new_tile, 2, 0)
        new_tile = new_tile.astype('float32')
        f.write("Output tile shape {} type {}\n".format(new_tile.shape, new_tile.dtype))
    elif formato == "npy":
        f.write("Image format {}\n".format(formato))
        new_tile = np.load(os.path.join(tile_path, tile_name))
        f.write("Input tile shape {} type {}\n".format(new_tile.shape, new_tile.dtype))
        new_tile = new_tile.astype('float32')
        f.write("Output tile shape {} type {}\n".format(new_tile.shape, new_tile.dtype))
    else:
        f.write("Image format TILE FORMAT INCORRECT\n")
        raise NotImplementedError("TILE FORMAT INCORRECT")

    if action == "tif2npy":
        f.write("Run tif2npy\n")
        new_tile[[0, 1, 2, 3, 4]] = new_tile[[2, 1, 0, 3, 4]]
        u, c = np.unique(new_tile, return_counts=True)
        f.write("Input tile shape {} type {} unique {} count {}\n".format(new_tile.shape, new_tile.dtype, u, c))
        # new_tile, mean, std, center, mx = normalizer(new_tile, f, center, center)
        vect = np.load(os.path.join(tile_path, 'norm.npy'))
        mean = vect[0]
        std = vect[1]
        new_tile = simple_normalizer(new_tile, f, mean, std)
        f.write("Norm param mean {} std {}\n".format(mean, std))
        f.write("Saved norm tile shape {} type {} min {} max {}\n".format(new_tile.shape, new_tile.dtype,
                                                                          np.min(new_tile), np.max(new_tile)))
        np.save(os.path.join(dest_path, name + "_" + tile_name.split(".")[0]), new_tile, allow_pickle=True)
        '''np.save(os.path.join(dest_path, name + "_mean_" + tile_name.split(".")[0]), mean, allow_pickle=True)
        np.save(os.path.join(dest_path, name + "_std_" + tile_name.split(".")[0]), std, allow_pickle=True)
        np.save(os.path.join(dest_path, name + "_center_" + tile_name.split(".")[0]), center, allow_pickle=True)
        np.save(os.path.join(dest_path, name + "_max_" + tile_name.split(".")[0]), mx, allow_pickle=True)'''
    elif action == "SAR_feature":
        f.write("Run real_SAR_feature_extractor\n")
        u, c = np.unique(new_tile, return_counts=True)
        f.write("Input tile shape {} type {} unique {} count {}\n".format(new_tile.shape, new_tile.dtype, u, c))
        new_tile = real_SAR_feature_extractor(new_tile, f, filter_type, filter_kernel)
        # new_tile, mean, std, center, mx = normalizer(new_tile, f, center, center)
        vect = np.load(os.path.join(tile_path, 'norm.npy'))
        mean = vect[0]
        std = vect[1]
        new_tile = simple_normalizer(new_tile, f, mean, std)
        f.write("Norm param mean {} std {}\n".format(mean, std))
        f.write("Saved norm tile shape {} type {} min {} max {}\n".format(new_tile.shape, new_tile.dtype,
                                                                          np.min(new_tile), np.max(new_tile)))
        np.save(os.path.join(dest_path, name + "_" + tile_name.split(".")[0]), new_tile, allow_pickle=True)
        '''np.save(os.path.join(dest_path, name + "_mean_" + tile_name.split(".")[0]), mean, allow_pickle=True)
        np.save(os.path.join(dest_path, name + "_std_" + tile_name.split(".")[0]), std, allow_pickle=True)
        np.save(os.path.join(dest_path, name + "_center_" + tile_name.split(".")[0]), center, allow_pickle=True)
        np.save(os.path.join(dest_path, name + "_max_" + tile_name.split(".")[0]), mx, allow_pickle=True)'''

    f.write("Execution time = {:.2f} s".format(time.time() - t))
    f.close()


def real_SAR_feature_extractor(raw_data, f, filer_type="box", filter_kernel=3):
    """
    This function got an np array of 4 band [{R_hh,I_hh,R_hv,I_hv},w,h] and return an an array of shape [5,w,h]
    where each pixel is composed of 5 real value calculated as in EUSAR paper
    To process [4,14000,9000] of type float32 it requires around 30 minutes
    Gaussian filter param
    - sigma < 0.5 -> 3x3
    - 0.5 <= sigma < 0.8334 -> 5x5
    - 0.8334 <= sigma < 1.17 -> 7x7
    - 1.17 <= sigma < 1.45 -> 9x9
    - sigma >= 1.45 -> 11x11
    Box filter only takes ikernel size 7 = 7x7 ecc
    :param raw_data: input data to be processed
    :param f:
    :param filer_type: {"gauss", "box"}
    :param filter_kernel:
    :return:
    """
    f.write("Data shape {} type {}\n".format(raw_data.shape, raw_data.dtype))

    # image dim
    w = raw_data.shape[1]
    h = raw_data.shape[2]

    # create equivalent complex vector
    complex_data = np.zeros((2, w, h), dtype=np.complex64)
    # fulfil complex vector
    complex_data[0].real = raw_data[0]
    complex_data[0].imag = raw_data[1]
    complex_data[1].real = raw_data[2]
    complex_data[1].imag = raw_data[3]

    f.write("Complex_data shape {} type {}\n".format(complex_data.shape, complex_data.dtype))

    # init vector for single pixel cov matrix
    s = np.zeros((1, 2), dtype=np.complex64)

    # initialize output feature vector
    data = np.zeros((5, w, h), dtype=np.float32)

    # initialize vector for all pixell covariance
    covariance = np.zeros((4, w, h), dtype=np.float32)

    # pass each pixel
    for i in range(0, w):
        for j in range(0, h):
            s[0, :] = complex_data[:, i, j]
            s_conj = np.conjugate(s)
            temp = s * s_conj.T
            c11_real = temp[0, 0].real
            c22_real = temp[1, 1].real
            c12 = temp[0, 1]
            covariance[:, i, j] = np.array((c11_real, c22_real, c12.real, c12.imag))


    # filter
    if filer_type == "box":
        for i in range(covariance.shape[0]):
            covariance[i] = ndimage.uniform_filter(covariance[i], filter_kernel, mode="wrap")
    else:
        covariance = ndimage.gaussian_filter(covariance, filter_kernel, truncate=3)

    f.write("Filter type applied {} kernell dim {}\n".format(filer_type, filter_kernel))
    # Calculate C12 abs
    # Init complex vector
    complex_c12 = np.zeros((w, h), dtype=np.complex64)
    # Fulfil complex vector
    complex_c12.real = covariance[2]
    complex_c12.imag = covariance[3]

    c12_abs = np.abs(complex_c12)

    # removes zeros
    covariance = np.where(covariance <= 0.0, 1e-06, covariance)
    c12_abs = np.where(c12_abs <= 0.0, 1e-06, c12_abs)

    # create output feature vector
    data[0] = np.log10(covariance[0])
    data[1] = np.log10(covariance[1])
    data[2] = np.log10(c12_abs)
    data[3] = covariance[2] / c12_abs
    data[4] = covariance[3] / c12_abs

    f.write("Return data array with 5 polsar real feature as EUSAR paper\n")

    return data


def print_values(data, comp=None):

    for i in range(2):
        for k in range(data.shape[0]):
            if comp is not None:
                print("{}-{}-{}-{}".format(k, i,i ,data[k, i, i]))
                if k<2:
                    print("{}-{}-{}-{}".format(k, i,i ,comp[k, i, i]))
            else:
                print("{}-{}-{}-{}".format(k, i,i ,data[k, i, i]))
        print()



