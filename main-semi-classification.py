import warnings

import numpy as np

from torch.utils.data import Dataset
import argparse
# from dataloader import load_data
import os
import torch.nn.functional as F
import torch
from tqdm import tqdm
# from trainers.DGMF_trainer import train as DGMF_train
# from trainers.DGMF_trainer_time import train as DGMF_train
from trainers.DGMF_trainer_time_final import train as DGMF_train

# import sys
# parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# sys.path.append(parent_dir)

from Utils import *

def save_res_log(metrics, metric_final, file_seed, mask='metrics'):
    print(mask, file=file_seed)
    save_res_seed(metrics, file_seed, args.seed)
    for k, v in metrics.items():
        metric_final[k].append(np.mean(v))

def main_mv(args):
    # model_trainer
    if args.model == 'DGMF':
        train = DGMF_train
    else:
        raise ValueError("False model!")

    # save_log
    res_dir = f'exp/result/{args.dataset}'
    file_seed, file_final, file_detail, weights_path, emb_path = get_logfile(res_dir, args.dir_h)
    metric_final = {'acc': [], 'p': [], 'r': [], 'f1': [], 'auc': [], 'time': []}
    for i, se in enumerate(args.all_seed):
        args.seed = se
        save_args(args, file_final)
        set_seed(args.seed)
        torch.cuda.empty_cache()
        metric_seed = train(args, weights_path, emb_path)
        save_res_log(metric_seed, metric_final, file_seed)
    save_res(metric_final, file_final)

def parameter_parser():
    parser = argparse.ArgumentParser(description='multi-model')
    subparsers = parser.add_subparsers(dest='model', required=False)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--description", type=str, default="---------")
    parent_parser.add_argument("--dir_h", type=str, default="debug")
    parent_parser.add_argument("--isConfig", action='store_true', default=True)
    parent_parser.add_argument("--train_detail_dir", type=str, default="online", help="Device: cuda:num or cpu")
    parent_parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parent_parser.add_argument("--n_repeated", type=int, default=1, help="Number of repeated times. Default is 10.")
    parent_parser.add_argument("--all_seed", type=int, default=[1], help="Random seed.")
    parent_parser.add_argument('--dataset', default='BBCsports',
                               choices=['BBCnews', 'BBCsports', 'NGs', 'Citeseer', '3sources', 'MSRC-v1', 'ALOI', 'animals',
                                 'Out_Scene', '100leaves', 'HW', 'MNIST', 'GRAZ02', 'Youtube', 'MNIST10k',
                                 'Reuters', 'Wikipedia', 'Caltech101-7', 'Caltech102', 'ORL', 'NUS-WIDE',
                                 'NoisyMNIST_15000', 'iaprtc12', 'YaleB_Extended'])
    parent_parser.add_argument("--workers", default=1)
    parent_parser.add_argument('--batch_size', default=200, type=int)
    # Multi-View
    parent_parser.add_argument("--add_conflict", default=False)
    parent_parser.add_argument("--add_Noise", default=False)

    parser_dgmf = subparsers.add_parser("DGMF", parents=[parent_parser])
    parser_dgmf.add_argument("--use_gmm", type=bool, default=True)
    parser_dgmf.add_argument("--use_shsp", type=bool, default=True)

    parser_dgmf.add_argument("--learning_rate", default=0.003)
    parser_dgmf.add_argument("--weight_decay", default=0.001)
    parser_dgmf.add_argument("--pre_epoch", type=int, default=100, help="Number of training epochs. Default is 200.")
    parser_dgmf.add_argument("--num_epoch", type=int, default=190, help="Number of training epochs. Default is 200.")
    parser_dgmf.add_argument("--alpha_a", type=float, default=0.4)
    parser_dgmf.add_argument("--knns", type=int, default=30, help="Number of k nearest neighbors")
    parser_dgmf.add_argument("--common_neighbors", type=int, default=2,
                             help="Number of common neighbors (when using pruning strategy 2)")
    parser_dgmf.add_argument("--pr1", action='store_true', default=False, help="Using prunning strategy 1 or not")
    parser_dgmf.add_argument("--pr2", action='store_true', default=False, help="Using prunning strategy 2 or not")
    parser_dgmf.add_argument("--ratio", type=float, default=0.1, help="Ratio of labeled samples")
    parser_dgmf.add_argument("--dropout", type=float, default=0.5)
    parser_dgmf.add_argument("--K", type=float, default=3)
    parser_dgmf.add_argument("--l1", type=float, default=1)
    parser_dgmf.add_argument("--l2", type=float, default=0.001)
    parser_dgmf.add_argument("--l3", type=float, default=0.0001)
    parser_dgmf.add_argument('--residual', type=bool, nargs='?', default=True, help='Use residual')
    parser_dgmf.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                             help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser_dgmf.add_argument('--v_drop', type=float, nargs='?', default=0.5,
                             help='Views attention Dropout (1 - keep probability).')
    parser_dgmf.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
    parser_dgmf.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each GAT layer')
    parser_dgmf.add_argument('--view_head_config', type=str, nargs='?', default='16',
                             help='Encoder layer config: # attention heads in each Views layer')
    parser_dgmf.add_argument('--view_layer_config', type=str, nargs='?', default='128',
                             help='Encoder layer config: # units in each Views layer')

    args, remaining_args = parser.parse_known_args()
    return args

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    args.device = device

    main_mv(args)