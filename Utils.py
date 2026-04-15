import configparser
import datetime
import json
import os
import sys
import ast

import numpy as np
import random
import torch
from texttable import Texttable
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, accuracy_score

def norm_2(x, y):
    return 0.5 * (torch.norm(x-y) ** 2)

# Config

def load_config(args, config_path):
    '''
    :param args: set args.isConfig(bool) 所有args的key必须小写,value改成对应类型如float-----> 1 to 1.
    :param config_path: './config.ini'
    :return:
    '''
    if args.isConfig:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(config_path)
        if config.has_section(args.dataset):
            for key, value in config.items(args.dataset):
                if hasattr(args, key):
                    current_value = getattr(args, key)
                    if isinstance(current_value, bool):
                        setattr(args, key, value.lower() in ['true', '1', 'yes'])
                    elif isinstance(current_value, float):
                        setattr(args, key, float(value))
                    elif isinstance(current_value, int):
                        setattr(args, key, int(value))
                    elif isinstance(current_value, list):
                        setattr(args, key, ast.literal_eval(value))
                    else:
                        setattr(args, key, value)
                else:
                    print(f"Warning: {key} in config not found in args.")
        else:
            raise ValueError(f"Dataset section '{args.dataset}' not found in the configuration file.")


def get_device(device:str):
    '''
    :param device: '0'
    :return:
    '''
    device = torch.device('cpu' if device == 'cpu' else 'cuda:' + device)
    return device


def set_root(dir:str='..'):
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir)
    os.chdir(root_path)


def set_seed(seed):
    '''
    :param seed:
    :return:
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def save_embeddings(emb_dir, seed, emb, y):
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    emb_dir = os.path.sep.join([emb_dir, f'seed_{seed}'])
    if os.path.exists(emb_dir) is False:
        os.makedirs(emb_dir)
    np.save(os.path.sep.join([emb_dir, f'embeddings.npy']), emb)
    np.save(os.path.sep.join([emb_dir, f'label.npy']), y)

def save_model(model, weights_path, seed):
    state = model.state_dict()
    emb_dir = os.path.sep.join([weights_path, f'seed_{seed}'])
    if os.path.exists(emb_dir) is False:
        os.makedirs(emb_dir)
    torch.save(state, weights_path + f'/seed_{seed}' + '/best_model.pth')
    # print("Saved best model")

def load_model(model, weights_path, seed):
    checkpoint = torch.load(weights_path + f'/seed_{seed}' + '/best_model.pth')
    model.load_state_dict(checkpoint)
    print("Loaded best model")
    return model



def save_args(args, file_final):
    args = vars(args)
    # keys = sorted(args.keys())
    keys = args.keys()
    t = Texttable(max_width=0)
    t.set_cols_width([30, 50])
    t.set_cols_dtype(['t', 't'])
    t.set_cols_align(['l', 'l'])
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", "_").capitalize(), str(args[k])] for k in keys])
    print(t.draw())
    print(t.draw(), file=file_final)
    print(f"---------------------------Max-Result----------------------------", file=file_final)
    file_final.flush()

def save_args_json(args, file_final):
    args_dict = vars(args)
    print(json.dumps(args_dict, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(args_dict, indent=2, ensure_ascii=False, default=str), file=file_final)
    print("---------------------------Max-Result----------------------------", file=file_final)
    file_final.flush()

def get_logfile(res_dir, exp_type=None, test=False):
    '''
    :param res_dir: f'exp/result/{args.model}/{args.dataset}'
    :param exp_type: args.dir_h
    :return:
    '''

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if exp_type is not None:
        log_path = os.path.join(res_dir, exp_type, current_time)
    else:
        log_path = os.path.join(res_dir, current_time)
    if os.path.exists(log_path) is False:
        os.makedirs(log_path)
    weights_path = log_path + "/weights"
    emb_path = log_path + "/embeddings"
    if not test:
        if os.path.exists(weights_path) is False:
            os.makedirs(weights_path)
        if os.path.exists(emb_path) is False:
            os.makedirs(emb_path)
    file_detail = open("{}/test_detail.log".format(log_path), 'a+')
    file_seed = open("{}/test_seed.log".format(log_path), 'a+')
    file_final = open("{}/test.log".format(log_path), 'a+')

    print(datetime.datetime.now(), file=file_detail)
    print(datetime.datetime.now(), file=file_seed)
    print(datetime.datetime.now(), file=file_final)
    file_seed.flush()
    file_final.flush()
    file_detail.flush()

    return file_seed, file_final, file_detail, weights_path, emb_path


def save_res_seed(matric:dict, file_seed, seed=-1, epoch=None, flag=None):
    '''
    :param matric: dict
    :param file_seed:
    :param seed:
    :param epoch: if n_repeat
    :return:
    '''
    res = f'Seed: {seed} '
    if epoch is not None:
        res += f'Epoch: {epoch} '

    if flag == 'clustering':
        # nmi, ari, acc, pur, fmi
        for key in matric.keys():
            if 'acc' in key.lower():
                res += f'ACC: {round(matric[key], 4)} '
            elif 'nmi' in key.lower():
                res += f'NMI: {round(matric[key], 4)} '
            elif 'ari' in key.lower():
                res += f'ARI: {round(matric[key], 4)} '
            elif 'pur' in key.lower():
                res += f'PUR: {round(matric[key], 4)} '
            elif 'fmi' in key.lower():
                res += f'FMI: {round(matric[key], 4)} '
            elif 'time' in key.lower():
                res += f'Time: {round(matric[key], 4)} '
    else:
        for key in matric.keys():
            if 'acc' in key.lower():
                res += f'ACC: {round(matric[key], 4)} '
            elif 'auc' in key.lower():
                res += f'AUC: {round(matric[key], 4)} '
            elif 'ap' in key.lower():
                res += f'AP: {round(matric[key], 4)} '
            elif any(substring in key.lower() for substring in ['recall', 'r']):
                res += f'Recall: {round(matric[key], 4)} '
            elif any(substring in key.lower() for substring in ['precision', 'p']):
                res += f'Precision: {round(matric[key], 4)} '
            elif 'f1_weighted' in key.lower():
                res += f'F1_weighted: {round(matric[key], 4)} '
            elif 'f1_macro' in key.lower():
                res += f'F1_macro: {round(matric[key], 4)} '
            elif 'f1' in key.lower():
                res += f'F1: {round(matric[key], 4)} '
            elif 'loss' in key.lower():
                res += f'{key}: {round(matric[key], 4)} '
            elif 'avg_train_time' in key.lower():
                res += f'avg_train_time: {round(matric[key], 4)} '
            elif 'avg_inference_time' in key.lower():
                res += f'avg_inference_time: {round(matric[key], 4)} '

    print(res)
    print(res, file=file_seed)
    file_seed.flush()


def save_res(matric: dict, file_final, seed=None, flag=None):
    '''
    :param matric: dict
    :param file_final:
    :param seed: All_seed
    :return:
    '''
    if seed is None:
        seed = [-1]
    res = f'Seed_Avg: {seed} '
    if flag == 'clustering':
        # nmi, ari, acc, pur, fmi
        for key in matric.keys():
            mean = round(np.mean(matric[key]), 4)
            std = round(np.std(matric[key]), 4)
            if 'acc' in key.lower():
                res += f'ACC: {mean}+{std} '
            elif 'nmi' in key.lower():
                res += f'NMI: {mean}+{std} '
            elif 'ari' in key.lower():
                res += f'ARI: {mean}+{std} '
            elif 'pur' in key.lower():
                res += f'PUR: {mean}+{std} '
            elif 'fmi' in key.lower():
                res += f'FMI: {mean}+{std} '
            elif 'time' in key.lower():
                res += f'Time: {mean}+{std} '
    else:
        for key in matric.keys():
            mean = round(np.mean(matric[key]), 4)
            std = round(np.std(matric[key]), 4)
            if 'acc' in key.lower():
                res += f'ACC: {mean}+{std} '
            elif 'auc' in key.lower():
                res += f'AUC: {mean}+{std} '
            elif 'ap' in key.lower():
                res += f'AP: {mean}+{std} '
            elif any(substring in key.lower() for substring in ['recall', 'r']):
                res += f'Recall: {mean}+{std} '
            elif any(substring in key.lower() for substring in ['precision', 'p']):
                res += f'Precision: {mean}+{std} '
            elif 'f1_weighted' in key.lower():
                res += f'F1_weighted: {mean}+{std} '
            elif 'f1_macro' in key.lower():
                res += f'F1_macro: {mean}+{std} '
            elif 'f1' in key.lower():
                res += f'F1: {mean}+{std} '
            elif 'time' in key.lower():
                res += f'Time: {mean}+{std} '

    print(res)
    print(res, file=file_final)
    file_final.flush()

def get_evaluation_results(labels_true, labels_pred, pred_score=None):
    '''
    :param labels_true: np.array
    :param labels_pred: np.array
    :param pred_score: np.array
    :return:
    '''
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_weighted = metrics.f1_score(labels_true, labels_pred, average='weighted')

    if pred_score is not None:
        if max(labels_true) >= 2:
            AUC = metrics.roc_auc_score(labels_true, pred_score, multi_class='ovo',average='macro')
            # AUC = -1
        else:
            AUC = metrics.roc_auc_score(labels_true, pred_score[:,1])
    else:
        AUC = -1
    return ACC, P, R, F1, F1_weighted, AUC

def get_evaluation_multilabel_results(labels_true, labels_pred, pred_score=None):
    '''
    :param labels_true: np.array
    :param labels_pred: np.array
    :param pred_score: np.array
    :return:
    '''

    def Accuracy(y_true, y_pred):
        count = 0
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for i in range(y_true.shape[0]):
            p = sum(np.logical_and(y_true[i], y_pred[i]))
            q = sum(np.logical_or(y_true[i], y_pred[i]))
            count += p / q
        return count / y_true.shape[0]

    Sub_ACC = metrics.accuracy_score(labels_true, labels_pred)
    ACC = Accuracy(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='samples')
    R = metrics.recall_score(labels_true, labels_pred, average='samples')
    F1_samples = metrics.f1_score(labels_true, labels_pred, average='samples')
    F1_macro = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_weighted = metrics.f1_score(labels_true, labels_pred, average='weighted')

    if pred_score is not None:
        if max(labels_true) >= 2:
            # AUC = metrics.roc_auc_score(labels_true, pred_score, multi_class='ovr')
            AUC = -1
        else:
            AUC = metrics.roc_auc_score(labels_true, pred_score[:,1])
    else:
        AUC = -1
    return ACC, P, R, F1_macro, F1_weighted, AUC

def get_evaluation_clustering_results(labels_true, labels_pred):

    def cluster_acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        u = linear_sum_assignment(w.max() - w)
        ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    def purity(y_true, y_pred):
        y_voted_labels = np.zeros(y_true.shape)
        labels = np.unique(y_true)
        ordered_labels = np.arange(labels.shape[0])
        for k in range(labels.shape[0]):
            y_true[y_true == labels[k]] = ordered_labels[k]
        labels = np.unique(y_true)
        bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

        for cluster in np.unique(y_pred):
            hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
            winner = np.argmax(hist)
            y_voted_labels[y_pred == cluster] = winner
        return accuracy_score(y_true, y_voted_labels)

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    acc = cluster_acc(labels_true, labels_pred)
    pur = purity(labels_true, labels_pred)
    fmi = metrics.fowlkes_mallows_score(labels_true, labels_pred)

    return nmi, ari, acc, pur, fmi


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count