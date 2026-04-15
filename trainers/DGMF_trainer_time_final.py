import copy
import warnings
import time

import numpy as np
from torch.utils.data import Dataset
import argparse
import os
import torch.nn.functional as F
import torch
from tqdm import tqdm
import scipy.sparse as ss
import sys
import torch_geometric as tg
from torch_geometric.data import Data
parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(parent_dir)

from models.DGMF_cross_view_final import *
from dataloaders.dataloader_graph import load_data
from Utils import *


def _sync_if_cuda(device):
    if isinstance(device, torch.device):
        is_cuda = device.type == 'cuda'
    else:
        is_cuda = 'cuda' in str(device)
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def save_embeddings(emb_dir, seed, emb, emb_sp, y, idx_unlabeled, conf=None):
    emb = emb[idx_unlabeled]
    y = y[idx_unlabeled]
    for l in range(len(emb_sp)):
        emb_sp[l] = emb_sp[l][idx_unlabeled]
        emb_sp[l] = emb_sp[l].cpu().detach().numpy()

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    emb_dir = os.path.sep.join([emb_dir, f'seed_{seed}'])
    if os.path.exists(emb_dir) is False:
        os.makedirs(emb_dir)
    np.save(os.path.sep.join([emb_dir, f'embeddings_sp.npy']), emb_sp)
    np.save(os.path.sep.join([emb_dir, f'embeddings.npy']), emb)
    np.save(os.path.sep.join([emb_dir, f'label.npy']), y)
    if conf is not None:
        if isinstance(conf, torch.Tensor):
            conf = conf.cpu().detach().numpy()
        np.save(os.path.sep.join([emb_dir, f'confidence.npy']), conf[idx_unlabeled])


def valid(args, model, feature_list, pyg_graphs, labels, idx_labeled, idx_unlabeled):
    model.eval()
    device = args.device
    with torch.no_grad():
        _sync_if_cuda(device)
        infer_start = time.perf_counter()
        loss_rec, loss_sp, loss_sh, z_sp, z_sh_cls, logit_sp, logit_sh, weights, total_energies, total_penalty = model_forward(
            args, model, feature_list, pyg_graphs, labels, idx_labeled
        )
        _sync_if_cuda(device)
        inference_time = time.perf_counter() - infer_start

        class_res = torch.max(logit_sp.softmax(dim=-1), logit_sh.softmax(dim=-1))
        class_sh = logit_sh.softmax(dim=-1)
        class_sp = logit_sp.softmax(dim=-1)
        conf_view = weights
        pred_labels = torch.argmax(class_res, 1).cpu().detach().numpy()
        pred_labels_sh = torch.argmax(class_sh, 1).cpu().detach().numpy()
        pred_labels_sp = torch.argmax(class_sp, 1).cpu().detach().numpy()
        ACC, P, R, F1, F1_weighted, AUC = get_evaluation_results(
            labels.cpu().detach().numpy()[idx_unlabeled],
            pred_labels[idx_unlabeled]
        )
        ACC_sh, P_sh, R_sh, F1_sh, _, _ = get_evaluation_results(
            labels.cpu().detach().numpy()[idx_unlabeled],
            pred_labels_sh[idx_unlabeled]
        )

        ACC_sp, P_sp, R_sp, F1_sp, _, _ = get_evaluation_results(
            labels.cpu().detach().numpy()[idx_unlabeled],
            pred_labels_sp[idx_unlabeled]
        )
        metrics = {'acc': ACC, 'p': P, 'r': R, 'f1': F1}
        metrics_sh = {'acc': ACC_sh, 'p': P_sh, 'r': R_sh, 'f1': F1_sh}
        metrics_sp = {'acc': ACC_sp, 'p': P_sp, 'r': R_sp, 'f1': F1_sp}

    return z_sh_cls, z_sp, conf_view, metrics, metrics_sh, metrics_sp, inference_time


def model_forward(args, model, feature_list, pyg_graphs, labels, idx_labeled):
    x_hat, z_sp_cls, z_sh_cls, logit_sp, logit_sh, weights, total_energies, total_penalty, z_sp = model(pyg_graphs)
    loss_rec, loss_sp, loss_sh = all_loss(x_hat, feature_list, logit_sp, logit_sh, labels, idx_labeled)
    return loss_rec, loss_sp, loss_sh, z_sp, z_sh_cls, logit_sp, logit_sh, weights, total_energies, total_penalty


def train(args, weights_path, emb_path):
    def train_step(epoch):
        _sync_if_cuda(args.device)
        train_start = time.perf_counter()

        loss_rec, loss_sp, loss_sh, _, _, _, _, _, total_energies, total_penalty = model_forward(
            args, model, feature_list, pyg_graphs, labels, idx_labeled
        )
        loss = args.l1*loss_rec + 1000*loss_sp + 1000*loss_sh + args.l2*total_energies+ args.l3*total_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _sync_if_cuda(args.device)
        train_time = time.perf_counter() - train_start

        # print("train_time: ", train_time)

        loss_meter.update(loss.item())
        return loss_rec, loss_sp, loss_sh, total_energies, total_penalty, train_time

    feature_list, adj_list, adj_f_list, adj_hat_list, adj_wave_list, labels, idx_labeled, idx_unlabeled = load_data(args)
    device = args.device
    ss_g_list = [ss.coo_matrix(adj) for adj in adj_list]
    feature_list = [torch.Tensor(x).to(device) for x in feature_list]

    num_view = len(feature_list)
    input_dims = [feature_list[i].shape[1] for i in range(num_view)]

    if args.dataset in ['Reuters', 'MNIST10k']:
        Defeature = Decomposition(input_dims, 256).to(device)
        x_de = Defeature(feature_list)
        input_dims = []
        for i in range(num_view):
            input_dims.append(x_de[i].shape[1])
            feature_list[i] = x_de[i].detach()
        torch.cuda.empty_cache()

    pyg_graphs = []
    for feat, adj in zip(feature_list, ss_g_list):
        x = torch.Tensor(feat)
        edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pyg_graphs.append(data)
    pyg_graphs = [g.to(device) for g in pyg_graphs]

    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    hidden_dim = [64, 32, 16]

    model = DGMF(args, input_dims, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)

    loss_meter = AverageMeter()
    total_train_time = 0.0
    total_inference_time = 0.0
    best_inference_time = None

    with tqdm(total=args.num_epoch, ncols=210, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            loss_rec, loss_sp, loss_sh, total_energies, total_penalty, train_time = train_step(epoch)
            total_train_time += train_time

            z_sh_cls, z_sp, conf_view, metrics, metrics_sh, metrics_sp, inference_time = valid(
                args, model, feature_list, pyg_graphs, labels, idx_labeled, idx_unlabeled
            )
            total_inference_time += inference_time

            # print("inference_time: ", inference_time)

            avg_train_time = total_train_time / (epoch + 1)
            avg_inference_time = total_inference_time / (epoch + 1)
            pbar.set_postfix({
                'loss_rec': '{:.6f}'.format((loss_rec).item()),
                'loss_sp': '{:.6f}'.format((loss_sp).item()),
                'loss_sh': '{:.6f}'.format((loss_sh).item()),
                'energies': '{:.6f}'.format((total_energies).item()),
                'penalty': '{:.6f}'.format((total_penalty).item()),
                'ACC': '{:.2f}'.format(metrics['acc'] * 100),
                'F1': '{:.2f}'.format(metrics['f1'] * 100),
                'avg_train_s': '{:.4f}'.format(avg_train_time),
                'infer_s': '{:.4f}'.format(inference_time)
            })
            pbar.update(1)

    result_metrics = metrics
    result_metrics_sh = metrics_sh
    result_metrics_sp = metrics_sp

    save_embeddings(emb_path, args.seed, z_sh_cls, z_sp, labels, idx_unlabeled, conf_view)
    save_model(model, weights_path, args.seed)

    avg_train_time_per_epoch = total_train_time / max(args.num_epoch, 1)
    avg_inference_time = total_inference_time / max(args.num_epoch, 1)

    print("------------------------")
    print("Average training time / epoch: {:.6f} s".format(avg_train_time_per_epoch))
    print("Average inference time: {:.6f} s".format(avg_inference_time))


    if best_inference_time is not None:
        print("Inference time of best checkpoint epoch: {:.6f} s".format(best_inference_time))
    print("------------------------")

    print("------------------------")
    print("ACC_sh:   {:.2f}".format(result_metrics_sh['acc'] * 100))
    print("P_sh :   {:.2f}".format(result_metrics_sh['p'] * 100))
    print("R_sh :   {:.2f}".format(result_metrics_sh['r'] * 100))
    print("F1_sh :   {:.2f}".format(result_metrics_sh['f1'] * 100))
    print("------------------------")

    print("------------------------")
    print("ACC_sp:   {:.2f}".format(result_metrics_sp['acc'] * 100))
    print("P_sp :   {:.2f}".format(result_metrics_sp['p'] * 100))
    print("R_sp :   {:.2f}".format(result_metrics_sp['r'] * 100))
    print("F1_sp :   {:.2f}".format(result_metrics_sp['f1'] * 100))
    print("------------------------")

    return result_metrics

