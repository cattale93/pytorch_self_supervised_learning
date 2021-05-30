import os
import pickle as pkl
import torch
import random
from Lib.utils.generic.generic_utils import set_rand_seed

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This function set up parameters based on the arguments passed
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def config_routine(args):
    """
    This routine is used to set up folders, save some status.py data and ask for some info when setting up a network
    both for training or for evaluation
    :param args: arguments passed when launching script
    :return:
    """
    if args.restore_training:
        temp_epoch = args.start_from_epoch
        temp_path = args.restoring_rep_path
        args = pkl.load(open(os.path.join(args.global_path, args.restoring_rep_path, 'args.pkl'), "rb"))
        args.start_from_epoch = temp_epoch
        args.restoring_rep_path = temp_path
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print('Model path = {}'.format(os.path.join(args.global_path, args.restoring_rep_path)))
        print('Resuming epoch = {}'.format(args.start_from_epoch))
        print('Continuing exp {}'.format(args.experiment_name))
        temp = input('Correct?(y/n) ')
        if 'y' in temp:
            print("Let's go...")
        else:
            raise NotImplementedError("INCORRECT INIT")
    else:
        if args.experiment_name == "":
            name = input('Experiment name? (DESCRIPTION)')
            args.experiment_name = name
        # create required folder
        try:
            list_dir = os.listdir(os.path.join(args.global_path, args.log_dir))
            list_dir = list(filter(lambda x: '.' not in x, list_dir))
            id_list = []
            if list_dir:
                for i in list_dir:
                    id_list.append(int(i.split('_')[0]))
                unique_id = max(id_list) + 10
            else:
                unique_id = 10

            args.data_dir_train = os.path.join(args.global_path, args.data_dir_train)
            args.data_dir_test = os.path.join(args.global_path, args.data_dir_test)
            args.data_dir_val = os.path.join(args.global_path, args.data_dir_val)
            args.log_dir = os.path.join(args.global_path, args.log_dir)
            args.pretrained_GAN = os.path.join(args.global_path, args.pretrained_GAN)
            args.log_dir = os.path.join(args.log_dir, str(unique_id) + '_' + args.experiment_name)
            print('Created ' + str(unique_id) + '_' + args.experiment_name)
        except:
            args.log_dir = os.path.join(args.global_path, args.log_dir)
            args.log_dir = os.path.join(args.log_dir, args.experiment_name)

        if args.sar_c != args.optical_c:
            args.lambda_identity = 0.0

        file = open(os.path.join(args.data_dir_train, '1_log.txt'), 'r')
        lines = file.readlines()
        values = lines[9].split(',')
        dim1 = values[1].split(')')[0]
        dim2 = values[2].split(')')[0]
        dim1 = int(dim1)
        dim2 = int(dim2)
        args.train_size = [dim1, dim2]

        file = open(os.path.join(args.data_dir_test, '1_log.txt'), 'r')
        lines = file.readlines()
        values = lines[9].split(',')
        dim1 = values[1].split(')')[0]
        dim2 = values[2].split(')')[0]
        dim1 = int(dim1)
        dim2 = int(dim2)
        args.test_size = [dim1, dim2]

        if 'BERLIN' in args.data_dir_train:
            args.sar_c = 2
            args.N_classes = 10

        if '32' in args.data_dir_train:
            args.patch_size = 32
        elif '128' in args.data_dir_train:
            args.patch_size = 128
        elif '256' in args.data_dir_train:
            args.patch_size = 256
        else:
            args.patch_size = 192

        if args.restoring_rep_path is not None:
            args.restoring_rep_path = os.path.join(args.global_path, args.restoring_rep_path, "checkpoints")

        os.mkdir(args.log_dir)
        # add new argument checkpoint_dir
        args.checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
        os.mkdir(args.checkpoint_dir)
        # add new argument tb_dir for tensorboard
        args.tb_dir = os.path.join(args.log_dir, "tb")
        os.mkdir(args.tb_dir)

        # if seed not available
        if args.seed is None:
            # generate seed
            args.seed = set_rand_seed()
            print("Random Seed: ", args.seed)
        # Set seed og random generators
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # create a log file with all options specified
        f = open(os.path.join(args.log_dir, "param.txt"), "a")
        text_line = "=" * 20 + "CONFIG" + "=" * 20 + '\n'
        f.write(text_line)
        for arg in vars(args):
            text_line = '{0:20}  {1}\n'.format(arg, getattr(args, arg))
            f.write(text_line)
        f.close()
        # save config variable
        pkl.dump(args, open(os.path.join(args.checkpoint_dir, "args.pkl"), "wb"))

    # check gpu
    if torch.cuda.is_available():
        print("GPU devices found: {}".format(torch.cuda.device_count()))
    else:
        raise NotImplementedError("GPU PROBLEM")

    # Create a file in the number of total epoch is stored
    # This file can be used to correctly stop the execution at a different number of epoch
    f = open(os.path.join(args.log_dir, "q.txt"), "w")
    val = 'epoch=' + str(args.tot_epochs)
    f.write(val)
    f.close()

    return args
