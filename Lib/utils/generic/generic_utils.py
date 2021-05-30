import time
import random
import numpy as np
from colour import Color
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: Here there are some general purpose functions
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def start():
    """
    Return the current time, works in couple with stop
    :return:
    """
    tic = time.time()
    return tic


def stop(toc, process_name=''):
    """
    Works is couple with start and take the returned actual time captured by start to calculate and print the difference
    :param toc:
    :param process_name: optional string appended before printing the time to give some detail
    :return:
    """
    print("{} execution time = {:.5f} s".format(process_name, time.time() - toc))


def set_rand_seed():
    """
    return a random value between 1 and 10000
    :return:
    """
    return random.randint(1, 10000)


def mean(arr, support=None):
    """
    Calculate the mean of an array
    :param arr: input list of data
    :return:
    """
    if support is not None:
        return sum(arr)/sum(support)
    else:
        return sum(arr)/len(arr)


def moving_average(data, win_dim=3, correct_init=False):
    """
    Calculate a moving mean. In moving mean first samples are not filterd typically because there are not enough sample.
    If use correct init, a tenth of the filtering window is copied in the first samples.
    :param data: input array of any type
    :param win_dim: dimension of filtering window
    :param correct_init: decide if to filter even firsts samples
    :return: averaged array
    """
    k = 10
    data = np.array(data).astype(float)
    win = []
    mov_data = []
    for value in data:
        if len(win) >= win_dim:
            win.pop(0)
            win.append(value)
        else:
            win.append(value)
        mov_data.append(np.mean(win))
    if correct_init:
        back_ward_mean = mov_data[int(win_dim/k):2*int(win_dim/k)]
        mov_data[0:int(win_dim/k)] = back_ward_mean
    return mov_data


def color_hex_list_generator(start_col, end_col, n_shades=5):
    """
    Create list of shaded colour
    :param start_col: starting color
    :param end_col: destination colour
    :param n_shades: number of colours
    :return: return list of colour
    """
    start_col = Color(start_col)
    colors = list(start_col.range_to(Color(end_col), n_shades))
    color_list = []
    for i in colors:
        color_list.append(i.hex)
    return color_list


def get_norm_param(data_path):
    """
    :param data_path: path to rdar or rgb folder
    :return: ['mean', 'std', 'center', 'max']
    """
    dir_list = sorted(os.listdir(data_path))
    param_name = []
    for x in dir_list:
        param_name.append(x.split('_')[1])
    param_name = list(filter(lambda x: '.' not in x, param_name))
    file_name = list(filter(lambda x: x.split('_')[1] in param_name, dir_list))
    temp = []
    for x in file_name:
        temp.append(np.load(os.path.join(data_path, x)))
    return temp[0], temp[1], temp[2], temp[3]