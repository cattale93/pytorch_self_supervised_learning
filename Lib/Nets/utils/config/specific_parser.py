# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file defines a function to overwrite parsed parameters. If specified is possible to overwrite passed
parameters. As a result is possible to define certain parameters in the initial par of scripts as done in the mains
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def specific_parser(parser, log=False, run_folder=None, mode=None, tot_epochs=None, restoring_rep_path=None,
                    start_from_epoch=None, pretrained_GAN=None, GAN_epoch=None, data_dir_train=None, data_dir_train2=None,
                    data_dir_test=None, data_dir_test2=None, images_log_freq=None, batch_size=None, batch_size_SN=None,
                    acc_log_freq=None, loss_log_freq=None, experiment_name=None, run_description=None, prc_train=None,
                    prc_test=None, prc_val=None, sar_c=None, optical_c=None, N_classes=None, patch_size=None, SN_log_freq=None,
                    save_model_freq=None, lambda_identity=None, D_training_ratio=None, lambda_A=None, loss_type=None,
                    lambda_gp=None, res_block_N=None, pool_prc_O=None, pool_prc_S=None, buff_dim=None, th_low=None, th_high=None,
                    pool=None, conditioned=None, dropping=None, th_b_h_ratio=None, th_b_l_ratio=None, th_b_h_pool=None,
                    th_b_l_pool=None, drop_prc=None, seed=None):
    """
    This is an intermediate layer between the general parser and the config routine to allow who use this code to easily
    access parameters and change them when building his experiment
    :param parser:
    :param log: decide if print or not
    :param run_folder: new value for run folder
    :param mode: train mode
    :param tot_epochs:
    :param restoring_rep_path:
    :param start_from_epoch:
    :param pretrained_GAN:
    :param GAN_epoch:
    :param data_dir_train:
    :param data_dir_train2:
    :param data_dir_test:
    :param data_dir_test2:
    :param images_log_freq:
    :param batch_size:
    :param batch_size_SN:
    :param acc_log_freq:
    :param loss_log_freq:
    :param experiment_name:
    :param run_description:
    :param prc_train:
    :param prc_test:
    :param prc_val:
    :param sar_c:
    :param optical_c:
    :param N_classes:
    :param patch_size:
    :param SN_log_freq:
    :param save_model_freq:
    :param lambda_identity:
    :param D_training_ratio:
    :param lambda_A:
    :param loss_type:
    :param lambda_gp:
    :param res_block_N:
    :param pool_prc_O:
    :param pool_prc_S:
    :param buff_dim:
    :param th_low:
    :param th_high:
    :param pool:
    :param conditioned:
    :param dropping:
    :param th_b_h_ratio:
    :param th_b_l_ratio:
    :param th_b_h_pool:
    :param th_b_l_pool:
    :param drop_prc:
    :return: args
    """
    args = parser.parse_args()
    print('SPECIFIC CONFIG')
    args.log_dir = update_arg(args.log_dir, run_folder, 'log_dir', log)
    args.tot_epochs = update_arg(args.tot_epochs, tot_epochs, 'tot_epochs', log)
    args.mode = update_arg(args.mode, mode, 'mode', log)
    args.restoring_rep_path = update_arg(args.restoring_rep_path, restoring_rep_path, 'restoring_rep_path', log)
    args.start_from_epoch = update_arg(args.start_from_epoch, start_from_epoch, 'start_from_epoch', log)
    args.pretrained_GAN = update_arg(args.pretrained_GAN, pretrained_GAN, 'pretrained_GAN', log)
    args.GAN_epoch = update_arg(args.GAN_epoch, GAN_epoch, 'GAN_epoch', log)
    args.data_dir_train = update_arg(args.data_dir_train, data_dir_train, 'data_dir_train', log)
    args.data_dir_train2 = update_arg(args.data_dir_train2, data_dir_train2, 'data_dir_train2', log)
    args.data_dir_test = update_arg(args.data_dir_test, data_dir_test, 'data_dir_test', log)
    args.data_dir_test2 = update_arg(args.data_dir_test2, data_dir_test2, 'data_dir_test2', log)
    args.images_log_freq = update_arg(args.images_log_freq, images_log_freq, 'images_log_freq', log)
    args.batch_size = update_arg(args.batch_size, batch_size, 'batch_size', log)
    args.batch_size_SN = update_arg(args.batch_size_SN, batch_size_SN, 'batch_size_SN', log)
    args.acc_log_freq = update_arg(args.acc_log_freq, acc_log_freq, 'acc_log_freq', log)
    args.loss_log_freq = update_arg(args.loss_log_freq, loss_log_freq, 'loss_log_freq', log)
    args.experiment_name = update_arg(args.experiment_name, experiment_name, 'experiment_name', log)
    args.run_description = update_arg(args.run_description, run_description, 'run_description', log)
    args.prc_train = update_arg(args.prc_train, prc_train, 'prc_train', log)
    args.prc_test = update_arg(args.prc_test, prc_test, 'prc_test', log)
    args.prc_val = update_arg(args.prc_val, prc_val, 'prc_val', log)
    args.sar_c = update_arg(args.sar_c, sar_c, 'sar_c', log)
    args.optical_c = update_arg(args.optical_c, optical_c, 'optical_c', log)
    args.N_classes = update_arg(args.N_classes, N_classes, 'N_classes', log)
    args.patch_size = update_arg(args.patch_size, patch_size, 'patch_size', log)
    args.SN_log_freq = update_arg(args.SN_log_freq, SN_log_freq, 'SN_log_freq', log)
    args.save_model_freq = update_arg(args.save_model_freq, save_model_freq, 'save_model_freq', log)
    args.lambda_identity = update_arg(args.lambda_identity, lambda_identity, 'lambda_identity', log)
    args.D_training_ratio = update_arg(args.D_training_ratio, D_training_ratio, 'D_training_ratio', log)
    args.lambda_A = update_arg(args.lambda_A, lambda_A, 'lambda_A', log)
    args.loss_type = update_arg(args.loss_type, loss_type, 'loss_type', log)
    args.lambda_gp = update_arg(args.lambda_gp, lambda_gp, 'lambda_gp', log)
    args.res_block_N = update_arg(args.res_block_N, res_block_N, 'res_block_N', log)
    args.pool_prc_O = update_arg(args.pool_prc_O, pool_prc_O, 'pool_prc_O', log)
    args.pool_prc_S = update_arg(args.pool_prc_S, pool_prc_S, 'pool_prc_S', log)
    args.buff_dim = update_arg(args.buff_dim, buff_dim, 'buff_dim', log)
    args.th_low = update_arg(args.th_low, th_low, 'th_low', log)
    args.th_high = update_arg(args.th_high, th_high, 'th_high', log)
    args.pool = update_arg(args.pool, pool, 'pool', log)
    args.conditioned = update_arg(args.conditioned, conditioned, 'conditioned', log)
    args.dropping = update_arg(args.dropping, dropping, 'dropping', log)
    args.th_b_h_ratio = update_arg(args.th_b_h_ratio, th_b_h_ratio, 'th_b_h_ratio', log)
    args.th_b_l_ratio = update_arg(args.th_b_l_ratio, th_b_l_ratio, 'th_b_l_ratio', log)
    args.th_b_h_pool = update_arg(args.th_b_h_pool, th_b_h_pool, 'th_b_h_pool', log)
    args.th_b_l_pool = update_arg(args.th_b_l_pool, th_b_l_pool, 'th_b_l_pool', log)
    args.drop_prc = update_arg(args.drop_prc, drop_prc, 'drop_prc', log)
    args.seed = update_arg(args.seed, seed, 'seed', log)
    return args


def update_arg(original, new_val, name, log=False):
    """
    Decide if update value or keep the original
    :param original:
    :param new_val:
    :param name: name of the variable
    :param log: decide if print or not
    :return:
    """
    if new_val is None:
        out_val = original
    else:
        out_val = new_val
    if log:
        print(' - ' + name + ' = {}'.format(out_val))

    return out_val
