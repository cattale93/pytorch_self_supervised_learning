import os
import torch
from torch.utils.data import Subset
from Lib.Nets.utils.generic.image2tensorboard import display_label_single_c, display_predictions
from random import randint, uniform
from PIL import Image
import numpy as np


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file defines some function which are common to all the network implemeted.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def breaker(args, epoch):
    """
    When execute the config routine a q.txt (q = quit) file is created where required number of epoch is written.
    By manually changing this parameter is possible to stop training at a different number of epochs.
    It is better to stop the network in this way so that each epoch is completely executed.
    This specific function read the value on the q file and return a bool which states if or not is equal to the actual
     number of epochs
    :param args: running parameters
    :param epoch: actual epoch
    :return:
    """
    f = open(os.path.join(args.global_path, args.log_dir, 'q.txt'), "r")
    epoch_end = int(f.readline().split('=')[-1])
    f.close()
    return epoch_end == epoch


def set_requires_grad(model, requires_grad=False, N=None):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    :param model: model to who turn off gradient
    :param requires_grad: True to request gradient
    :return:
    """
    if N is None:
        for param in model.parameters():
            param.requires_grad = requires_grad
        if requires_grad:
            model.train()
        else:
            model.eval()
    else:
        for i, param in enumerate(model.parameters()):
            if i < N:
                param.requires_grad = requires_grad


def print_param(model):
    """
    Take a model ad input and print all parameter data, it is a debugging function
    :param model:
    :return:
    """
    i = 0
    for name, param in model.named_parameters():
        print(name, param.data, param.requires_grad)
        i = i + 1
    print(i)


def get_subset(dataset, prc, end=False):
    """
    Return a loader definition with the right prc of patches
    Random disable so that all net use exactly same validation set
    :param dataset: dataset from where create the loader
    :param prc: net options
    :param rand:
    :return:
    """
    if prc >= 1:
        return dataset
    length = len(dataset)
    n_sample = int(length * prc)
    if end:
        subset = Subset(dataset, range(length - n_sample, length))
        print('Range of prc data {} is [{}, {}]'.format(prc, length - n_sample, length))
    else:
        start = 0
        subset = Subset(dataset, range(start, start + n_sample))
        print('Range of prc data {} is [{}, {}]'.format(prc, start, start + n_sample))
    return subset


def calculate_accuracy(Net, dataset, writer, global_step, name, epoch, posx, posy, size, BL=False):
    """
    Classify dataset and calculate accuracy
    :param Net:
    :param dataset: dataset to use to calculate accuracy
    :param writer: pointer to tb
    :param global_step:
    :param name:
    :param epoch:
    :return:
    """
    # iterate all or partially the dataset and for each sample calculate the accuracy
    # bar = tqdm(enumerate(dataset), total=len(dataset))
    map_list = np.zeros((len(dataset.dataset), 3, Net.opt.patch_size, Net.opt.patch_size),
                                      dtype=np.float32)

    set_requires_grad(Net.netG_S2O, False)
    set_requires_grad(Net.SegNet, False)
    for i, data in enumerate(dataset):
        label = data['label'].to(Net.device)
        radar = data['radar'].to(Net.device)
        patch_names = data['name']
        feature_map = Net.netG_S2O(radar)
        seg_map = Net.SegNet(feature_map)
        if name == 'Test':
            Net.Accuracy_test.update_acc(label, seg_map)
        else:
            Net.Accuracy_train.update_acc(label, seg_map)

        label_norm, mask = display_label_single_c(label)
        seg_map_norm = display_predictions(seg_map, True, mask)
        #writer.add_images(name + '/' + str(i) + "/Map", seg_map_norm, global_step=i)
        #writer.add_images(name + "/Labels", label_norm, global_step=i)

        img = seg_map_norm
        for k, n in enumerate(patch_names):
            temp = int(n.split('_')[1]) - 1
            map_list[temp] = img[k]

    set_requires_grad(Net.netG_S2O, BL)
    set_requires_grad(Net.SegNet, True)

    q_ps = int(Net.opt.patch_size / 4)
    ps_d = Net.opt.patch_size - q_ps
    tile = np.zeros((3, size[0], size[1]))
    for i in range(map_list.shape[0]):
        x = int(posx[i])
        y = int(posy[i])
        patch_o = map_list[i]
        tile[:, x + q_ps:x + ps_d, y + q_ps:y + ps_d] = patch_o[:, q_ps:ps_d, q_ps:ps_d]
    temp = tile[:, 800:7800, 1500:12000]
    temp = np.moveaxis(temp, 0, 2)
    temp = temp * 255
    temp = temp.astype(np.uint8)
    temp = Image.fromarray(temp, 'RGB')
    temp.save(os.path.join(Net.opt.tb_dir, str(epoch) + name + '_map.png'))

    # get the dictionary of the mean accuracies
    if name == 'Test':
        acc = Net.Accuracy_test.get_mean_dict()
        Net.Accuracy_test.reinit()
        # log the accuracy values
        Net.Logger_test.append_SN_acc(acc)
        # add accuracies to tensorboard
    else:
        acc = Net.Accuracy_train.get_mean_dict()
        Net.Accuracy_train.reinit()
        # log the accuracy values
        Net.Logger_train.append_SN_acc(acc)
        # add accuracies to tensorboard
    print('Accuracy = {}'.format(acc))
    writer.add_scalars(name + "/Accuracy", acc, global_step=global_step)


def cal_gp(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """
    Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU torch.device
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1, device=device)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
    return gradient_penalty, gradients


class List:
    """
    This class implements a buffer
    """
    def __init__(self, size=50):
        self.size = size
        # it is list of batch!
        self.data = []
        self.data.append(0.5)

    def push_and_pop(self, value):
        if len(self.data) < self.size:
            self.data.append(value)
        else:
            self.data.pop(0)
            self.data.append(value)

    def mean(self):
        return sum(self.data)/len(self.data)


def drop_channel(data, min_damping_coeff=0, p_th=0.7):
    n_channel = data.shape[1]
    channels = list(range(n_channel))
    N_of_drop_ch = randint(0, n_channel-1)
    prc = uniform(0, 1)
    if prc > p_th:
        for i in range(N_of_drop_ch):
            ch = randint(0, len(channels)-1)
            target_ch = channels.pop(ch)
            damping_coeff = round(uniform(0, min_damping_coeff), 1)
            data[:, target_ch] = data[:, target_ch] * damping_coeff
    return data
