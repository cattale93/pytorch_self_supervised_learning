import os
import torch
from PIL import ImageColor, Image
import numpy as np
from skimage import exposure
from scipy.ndimage import zoom


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file defines some function employed to manage images
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def display_label_multi_c(tensor):
    """
    Display one hot ancoded labels
    :param tensor: input tensor of label [NCHW]
    :return: a tensor of label ready for tensorboard
    """
    
    # calculate mask
    mask = get_mask_multi_c(tensor)
    tensor = torch.argmax(tensor, 1)
    # apply the mask, subtracting 5 where there are non classified pixel
    tensor = tensor - mask
    cmap = mycmap()
    # transform image values to color map
    tensor = cmap[tensor]

    # bring the last axis to the second position
    tensor = np.rollaxis(tensor, 3, 1)
    return tensor, mask


def display_label_single_c(tensor):
    """
    Displays labels
    :param tensor: input tensor of label [NHW]
    :return: a tensor of label ready for tensorboard
    """
    tensor = tensor.cpu().numpy()
    # calculate mask
    mask = get_mask_single_c(tensor)
    # apply the mask, subtracting 5 where there are non classified pixel
    tensor = tensor - mask
    cmap = mycmap()
    # transform image values to color map
    tensor = tensor.astype(int)
    tensor = cmap[tensor]
    # bring the last axis to the second position
    tensor = np.rollaxis(tensor, 3, 1)
    return tensor, mask


def display_predictions(tensor, mask_image=True, mask=0):
    """
    Display segmentation map
    :param tensor: input tensor of label [NCHW]
    :param mask_image: boolean to decide if to mask or not
    :param mask: mask array
    :return: a tensor of label ready for tensorboard
    """
    tensor = torch.argmax(tensor, 1)
    tensor = tensor.cpu().numpy()
    # apply the mask, subtracting 5 where there are non classified pixel
    if mask_image:
        tensor = np.where(mask == 256, -1, tensor)
    cmap = mycmap()
    # transform image values to color map
    tensor = tensor.astype(int)
    tensor = cmap[tensor]
    # bring the last axis to the second position
    tensor = np.rollaxis(tensor, 3, 1)

    return tensor


def display_input(tensor, mask_image=False, mask=0):
    """
    Display any kind of input
    :param tensor: tensor of rgb or radar [NCHW]
    :param mask_image: mask the image or not
    :param mask: mask array
    :return: return a tensor ready for tensorboard
    """
    if tensor.shape[1] == 2:
        temp = torch.zeros((tensor.shape[0], 3, tensor.shape[2], tensor.shape[2]))
        temp[:, 0] = tensor[:, 0]
        temp[:, 1] = tensor[:, 1]
        tensor = temp
    else:
        min_val = torch.min(tensor)
        # makes all value positive
        if min_val < 0:
            tensor = tensor - min_val
        max_val = torch.max(tensor)
        # check that value are meaningful and not all equal to zero
        if max_val != 0:
            tensor = torch.true_divide(tensor, max_val)
        tensor = tensor.cpu().detach().numpy()
        if mask_image:
            # [5] TODO: mascherare correttamente
            tensor = tensor * mask
        tensor = histogram_equalize(tensor)
    return tensor


def mycmap(norm_val=255, color_code="RGB"):
    """
    Define a colormap to be used when colouring maps
    :param norm_val: divide rgb vector with norm_val (1: no action, 255: normalize to 1)
    :param color_code: colour coding, RGBA o RGB
    :return: return an array of rgb or rgba codes
    """
    colours = np.array(["#0000ff",  # rosso  FF0000# water verde militare -> Forests
                        "#ff0000",  # lime 00FF00# grey -> strets
                        "#00ff00",  # blu 0000FF# lime -> fields
                        "#565656",  # azzurro 0000ff# red -> urban
                        "#145a32",  # verde militare 145A32# blu -> water
                        "#FFFF00",  # giallo
                        "#FF7000",  # arancio
                        "#FFFFFF",  # bianco
                        "#FF00FF",  # fucsia
                        "#767676",  # grigio
                        "#00CF84",  # verde acqua
                        "#ffffff"])  # nero #000000  # giallo
    # init array of colours
    cmap = np.zeros((len(colours), len(color_code)))
    # convert hex colour to rgb or rgba
    for i in range(len(colours)):
        cmap[i] = (np.array(ImageColor.getcolor(colours[i], color_code)))
    # normalize values between 0 and 1
    cmap = cmap/norm_val
    return cmap


def histogram_equalize(img):
    """
    Strech the images histogram between all channel to improve image quality
    :param img: image with any shape
    :return: same image with histogram streched
    """
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.array(np.interp(img, bin_centers, img_cdf))


def get_mask_multi_c(tensor):
    """

    :param tensor: input tensor of label
    :return: return a mask for that label
    """
    # create a mask with 1 where there are no classified pixel
    mask = torch.where(tensor == 0, torch.as_tensor(1), torch.as_tensor(0))

    mask = (mask[:, 0, :, :] * mask[:, 1, :, :] * mask[:, 2, :, :] * mask[:, 3, :, :] * mask[:, 4, :, :])*5


    return mask


def get_mask_single_c(tensor):
    """

    :param tensor: input tensor of label
    :return: return a mask for that label
    """
    # create a mask with 1 where there are no classified pixel
    mask = np.where(tensor == 255, 256, 0)
    return mask


def negate_mask(mask):
    """

    :param mask: mask of image
    :return: covert 0 to 1 and other value to 0
    """
    mask = torch.where(mask == 0, torch.as_tensor(1), torch.as_tensor(0))

    return mask


def norm(data, scale_factor=1, scale=False):
    """
    Brings data in the normal form for PIL saving
    :param data: image to be transformed
    :param scale_factor:
    :param scale:
    :return:
    """
    if data.shape[0] < 4:
        data = np.moveaxis(data, 0, -1)
    else:
        data = np.moveaxis(data, 0, 1)
    data = data * 255
    if scale:
        data = zoom(data, (scale_factor, scale_factor, 1), mode='nearest', prefilter=False)
    return data


def denorm(data, parameters_path, typ):
    """
    Denorm an image rescuing the original coding
    :param data: image to be decode
    :param parameters_path: path to coding parameters
    :param typ: if radar or rgb
    :return:
    """
    try:
        mx = np.load(os.path.join(parameters_path, '1_max_radar.npy'))
        for i in range(3):
            data[i, :, :] = data[i, :, :] * mx[i]
    except:
        pass
    center = np.load(os.path.join(parameters_path, '1_center_' + typ + '.npy'))
    std = np.load(os.path.join(parameters_path, '1_std_' + typ + '.npy'))
    mean = np.load(os.path.join(parameters_path, '1_mean_' + typ + '.npy'))
    for i in range(3):
        try:
            _ = len(center)
            data[i, :, :] = data[i, :, :] + center[i]
        except:
            pass
        data[i, :, :] = data[i, :, :] * std[i]
        data[i, :, :] = data[i, :, :] + mean[i]
    return data


def reconstruct_tile(name, ps, posx, posy, save_dir, size, epoch, data, rgb=True, parameter_path=None, data_s=None):
    media = 0.7678273
    q_ps = int(ps / 4)
    ps_d = ps - q_ps
    ch = data.shape[1]
    tile = torch.zeros((ch, size[0], size[1]))
    for i in range(data.shape[0]):
        x = int(posx[i])
        y = int(posy[i])
        patch_o = data[i]
        tile[:, x + q_ps:x + ps_d, y + q_ps:y + ps_d] = patch_o[:, q_ps:ps_d, q_ps:ps_d]
    #temp = np.array(tile[:, 800:7800, 1500:12000])
    #np.save(os.path.join(save_dir, str(epoch) + name + '.png'), temp)
    #tile_s = denorm(tile_s, os.path.join(parameter_path, 'radar'), 'radar')
    torch.save(tile, os.path.join(save_dir, str(epoch) + name + '.pt'))
    temp = tile.cpu().detach().numpy()
    temp = np.moveaxis(temp[0:3, 800:7800, 1500:12000], 0, -1)
    temp = temp - np.min(temp)
    if rgb:
        temp = np.log(temp + 1)
        temp = np.where(temp > 2.5 * media, 2.5 * media, temp)
        temp = temp / (2.5 * media)
    else:
        temp = temp / np.max(temp)
    temp = temp * 255
    temp = temp.astype(np.uint8)
    temp = Image.fromarray(temp, 'RGB')
    temp.save(os.path.join(save_dir, str(epoch) + name + '.png'))
