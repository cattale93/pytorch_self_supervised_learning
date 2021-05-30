# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file defines a parser to describe all the parameters emplyed in this work. This function allows to
pass parameter when launching the script.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def general_parser(parser):
    """
    This is the parser which collects all the parameters passed through the command to launch the script
    - experiment_name
    - run_description
    - val_freq
    - acc_log_freq
    - loss_log_freq
    - images_log_freq
    - SN_log_freq
    - save_model_freq
    - restore_training
    - start_from_epoch
    - restoring_rep_path
    - D_training_ratio
    - buff_dim
    - th_low
    - th_high
    - th_b_h_ratio
    - th_b_l_ratio
    - th_b_h_pool
    - th_b_l_pool
    - res_block_N
    - pool_prc_O
    - pool_prc_S
    - drop_prc
    - seed
    - tot_epochs
    - decay_epochs
    - patch_size
    - batch_size
    - sar_c
    - optical_c
    - N_classes
    - loss_type
    - lr
    - beta1
    - workers
    - dropout
    - lambda_S
    - lambda_O
    - lambda_identity
    - pool_size
    - pool
    - conditioned
    - pool
    - init_type
    - init_gain
    - global_path
    - data_dir_train
    - data_dir_train2
    - data_dir_test
    - data_dir_test2
    - data_dir_val
    - log_dir
    - train_set_prc
    - mode
    - lr_SN
    - weight_decay_SN
    - batch_size_SN
    """
    # general
    parser.add_argument('--experiment_name', type=str, default="",
                        help="experiment name. will be used in the path names for log- and save files")
    parser.add_argument('--run_description', type=str, default="",
                        help="describes the running")
    parser.add_argument('--val_freq', type=int, default=1000,
                        help='validation will be run every val_freq batches/optimization steps during training')
    parser.add_argument('--acc_log_freq', type=int, default=500,
                        help='model will be saved every acc_log_freq epochs during training')
    parser.add_argument('--loss_log_freq', type=int, default=10,
                        help='tensorboard logs will be written every loss_log_freq number of batches/optimization steps')
    parser.add_argument('--images_log_freq', type=int, default=1000,
                        help='tensorboard logs will be written every image_log_freq of batches/optimization steps')
    parser.add_argument('--SN_log_freq', type=int, default=5,
                        help='tensorboard logs will be written every image_log_freq of batches/optimization steps')
    parser.add_argument('--save_model_freq', type=int, default=5,
                        help='tensorboard logs will be written every image_log_freq of batches/optimization steps')
    parser.add_argument('--restore_training', type=bool, default=False,
                        help='restore a previous training')
    parser.add_argument('--start_from_epoch', type=int, default=0,
                        help='epoch from where start training')
    parser.add_argument('--restoring_rep_path', type=str, default=None,
                        help='restore a previous training')
    parser.add_argument('--D_training_ratio', type=int, default=5,
                        help='every D_training_ratio opt step of the generator 1 Discriminator opt is performed')
    parser.add_argument('--buff_dim', type=int, default=10000,
                        help='mean loss value on which decide if change Dration')
    parser.add_argument('--th_low', type=int, default=0.45,
                        help='D ratio low threshold')
    parser.add_argument('--th_high', type=int, default=0.55,
                        help='D ration high threshold')
    parser.add_argument('--th_b_h_ratio', type=int, default=100,
                        help='D ration high threshold')
    parser.add_argument('--th_b_l_ratio', type=int, default=2,
                        help='D ration high threshold')
    parser.add_argument('--th_b_h_pool', type=int, default=0.9,
                        help='D ration high threshold')
    parser.add_argument('--th_b_l_pool', type=int, default=0.4,
                        help='D ration high threshold')


    parser.add_argument('--res_block_N', type=int, default=9,
                        help='number of resblock ps=128 -> 6 ps= 256-> 9')
    parser.add_argument('--pool_prc_S', type=int, default=0.5,
                        help='prc to choose old generated patches or new one')
    parser.add_argument('--pool_prc_O', type=int, default=0.5,
                        help='prc to choose old generated patches or new one')
    parser.add_argument('--drop_prc', type=int, default=0.5,
                        help='prc to choose old generated patches or new one')

    # training hyperparameters
    parser.add_argument('--seed', type=int, default=1,
                        help='torch random seed')
    parser.add_argument('--tot_epochs', type=int, default=200,
                        help='number of training epochs (default: 100)')
    parser.add_argument("--decay_epochs", type=int, default=100,
                        help="when starts linearly decaying the learning rate to 0. (default:100)")
    parser.add_argument('--patch_size', type=int, default=32,
                        help='patch size for training and validation (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training and validation (default: 1)')
    parser.add_argument('--sar_c', type=int, default=5,
                        help='n of channel in sar images')
    parser.add_argument('--optical_c', type=int, default=4,
                        help='n of channel in optical images')
    parser.add_argument('--N_classes', type=int, default=5,
                        help='n of channel in optical images')
    parser.add_argument('--loss_type', type=str, default="lsgan",
                        help='[lsgan | wgan]')
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate. (default:0.0002)")
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 term for adam')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloading (default: 4)')
    parser.add_argument('--dropout', type=bool, default=True,
                        help='dropout for the generators: True for training, False for testing')
    parser.add_argument('--bias', type=bool, default=True,
                        help='bias for G and D: True for InstanceNorm2D as normalization func')
    parser.add_argument("--lambda_S", type=float, default=10,
                        help="weight for cycle loss (S -> O -> S)")
    parser.add_argument("--lambda_O", type=float, default=10,
                        help="weight for cycle loss (O -> S -> O)")
    parser.add_argument("--lambda_identity", type=float, default=0.5,
                        help="Scales the weight of the identity mapping loss.")
    parser.add_argument("--lambda_A", type=float, default=100,
                        help="Scales the weight of the regression loss.")
    parser.add_argument("--lambda_gp", type=float, default=10,
                        help="Scales the weight of the gradient penalty only for wgan loss.")
    parser.add_argument("--pool_size", type=float, default=50,
                        help="dim of the pool of images")
    parser.add_argument("--pool", type=bool, default=True,
                        help="if use pool or not")
    parser.add_argument("--conditioned", type=bool, default=False,
                        help="if use conditioned or not")
    parser.add_argument("--dropping", type=bool, default=False,
                        help="if use dropping or not")
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')

    # data
    parser.add_argument('--global_path', type=str, default="/home/ale/Documents/Python/13_Tesi_2/",
                        help='path to training dataset')
    parser.add_argument('--data_dir_train', type=str, default="Data/Train/EUSAR/32_box_double_norm",
                        help='path to training dataset')
    parser.add_argument('--data_dir_train2', type=str, default="Data/Train/EUSAR/32_box_double_norm",
                        help='path to training dataset')
    parser.add_argument('--data_dir_test', type=str, default="",
                        help='path to test dataset')
    parser.add_argument('--data_dir_test2', type=str, default="",
                        help='path to test dataset')
    parser.add_argument('--data_dir_val', type=str, default="Data/Train/EUSAR/32_box_double_norm",
                        help='path to validation dataset')
    parser.add_argument('--log_dir', type=str, default="Runs/Runs_CGAN/",
                        help='path to dir for code logs')
    parser.add_argument('--prc_train', type=int, default=1,
                        help='% of the train dataset')
    parser.add_argument('--prc_test', type=int, default=1,
                        help='% of the test dataset')
    parser.add_argument('--prc_val', type=int, default=1,
                        help='% of the val dataset')

    # SN param
    parser.add_argument('--mode', type=str, default='trainSN',
                        help='set up model mode [train | eval]')
    parser.add_argument('--lr_SN', type=float, default=0.01,
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--weight_decay_SN', type=float, default=5e-4,
                        help='weight-decay (default: 5e-4)')
    parser.add_argument('--batch_size_SN', type=int, default=32,
                        help='batch size for training and validation \
                              (default: 32)')
    parser.add_argument('--pretrained_GAN', type=str,
                        default='Runs/Runs_CGAN/4_2020-11-12_11-09-03_norm_data_first/norm_data_first_checkpoints',
                        help='restore a pretrained model')
    parser.add_argument('--GAN_epoch', type=int, default=None,
                        help='which epoch restore?')
    return parser
