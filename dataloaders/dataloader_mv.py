import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import h5py
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io
import torch
from sklearn.preprocessing import normalize

class BaseMultiViewDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X = None          # list of views
        self.Y = None          # labels
        self.num_views = None
        self.num_classes = None


    def __len__(self):
        return self.X[0].shape[0]

    def __getitem__(self, idx):
        views = [torch.from_numpy(self.X[v][idx]) for v in range(self.num_views)]
        label = torch.from_numpy(np.array(self.Y[idx])).long()
        index = torch.tensor(idx).long()
        return views, label, index

    def view_dim(self):
        return [x.shape[1] for x in self.X]

    # =================== post-processing ===================

    def postprocessing(self, test_index=None, addNoise=False, sigma=0, ratio_noise=0.5,
                       addConflict=False, ratio_conflict=0.5,
                       views_to_add=None):
        if addNoise:
            self.addNoise(test_index, ratio_noise, sigma, views_to_add)
        if addConflict:
            self.addConflict(test_index, ratio_conflict, views_to_add)

    def addNoise(self, test_index, ratio, sigma, views_to_noise=None):
        if test_index is None:
            index = np.arange(len(self.Y))
        else:
            index = test_index
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)

        for i in selects:
            if views_to_noise is not None:
                views = np.array(views_to_noise)
            else:
                views = np.random.choice(self.num_views,
                                          size=np.random.randint(1, self.num_views + 1),
                                          replace=False)
            for v in views:
                self.X[v][i] += np.random.normal(0, sigma, size=self.X[v][i].shape)

    def addConflict(self, test_index, ratio, views_to_conflict=None):
        # 每个类别选一个“模板样本”
        records = {}
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            records[c] = {v: self.X[v][i].copy() for v in range(self.num_views)}

        if test_index is None:
            index = np.arange(len(self.Y))
        else:
            index = test_index
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)

        for i in selects:
            conflict_class = (self.Y[i] + 1) % self.num_classes
            if views_to_conflict is not None:
                views = np.array(views_to_conflict)
            else:
                views = [np.random.randint(self.num_views)]

            for v in views:
                self.X[v][i] = records[conflict_class][v]


class BDGP(BaseMultiViewDataset):
    def __init__(self, path):
        # h5py.File(paths[args.dataset], 'r')
        super().__init__()
        mat = scipy.io.loadmat(path + 'BDGP.mat')
        X1 = mat['X1'].astype(np.float32)
        X2 = mat['X2'].astype(np.float32)
        Y = mat['Y'].transpose()

        self.X = [X1, X2]
        self.Y = Y.squeeze()
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        self.num_views = 2
        self.num_classes = self.Y.max() - self.Y.min() + 1


class Fashion(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        mat = scipy.io.loadmat(path + 'Fashion.mat')
        Y = mat['Y'].astype(np.int32).reshape(-1)
        X1 = mat['X1'].astype(np.float32).reshape(-1, 784)
        X2 = mat['X2'].astype(np.float32).reshape(-1, 784)
        X3 = mat['X3'].astype(np.float32).reshape(-1, 784)

        self.X = [X1, X2, X3]
        self.Y = Y
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        self.num_views = 3
        self.num_classes = self.Y.max() - self.Y.min() + 1


class HW(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        data = scipy.io.loadmat(path + 'HW.mat')
        # data['X'] 形如 (1, 6) 的 cell/对象数组：data['X'][0][v] 是第 v 个 view
        self.X = [normalize(data['X'][0][v].astype(np.float32)) for v in range(6)]
        # 标签：尽量变成一维 (n,)
        self.Y = data['Y']
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        self.num_views = 6
        self.num_classes = int(self.Y.max() - self.Y.min() + 1)


class hw6(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        data = scipy.io.loadmat(path + 'handwritten_6views.mat')
        scaler = MinMaxScaler((0, 1))

        view_train = [scaler.fit_transform(data[f'x{v+1}_train'].astype(np.float32)) for v in range(6)]
        view_test = [scaler.fit_transform(data[f'x{v+1}_test'].astype(np.float32)) for v in range(6)]

        Y_train = data['gt_train']
        Y_test = data['gt_test']
        self.Y = np.concatenate((Y_train, Y_test), axis=0)
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1

        self.X = [np.concatenate((view_train[v],view_test[v]),axis=0) for v in range(6)]
        self.num_views = 6
        self.num_classes = self.Y.max() - self.Y.min() + 1


class Caltech(BaseMultiViewDataset):
    def __init__(self, path, view):
        super().__init__()
        data = scipy.io.loadmat(path + 'Caltech-5V.mat')
        scaler = MinMaxScaler()

        self.Y = data['Y'].transpose()
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        view_dict = {2:[0,1], 3:[0,1,4], 4:[0,1,4,3], 5:[0,1,4,3,2]}
        self.X = [scaler.fit_transform(data[f'X{v+1}'].astype(np.float32)) for v in view_dict[view]]
        self.num_views = view
        self.num_classes = self.Y.max() - self.Y.min() + 1


class NGs(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        mat = scipy.io.loadmat(path + 'NGs.mat')
        x = mat['X']
        self.Y = mat['Y'].astype(np.int32).reshape(500, )
        X1 = x[0][0].astype(np.float32).reshape(-1, 2000)
        X2 = x[0][1].astype(np.float32).reshape(-1, 2000)
        X3 = x[0][2].astype(np.float32).reshape(-1, 2000)
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        self.X = [X1,X2,X3]
        self.num_views = 3
        self.num_classes = self.Y.max() - self.Y.min() + 1


class synthetic3d(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        mat = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = mat['Y'].astype(np.int32).reshape(600, )
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        x = mat['X']
        X1 = x[0][0].astype(np.float32).reshape(-1, 3)
        X2 = x[1][0].astype(np.float32).reshape(-1, 3)
        X3 = x[2][0].astype(np.float32).reshape(-1, 3)
        self.X = [X1,X2,X3]
        self.num_views = 3
        self.num_classes = self.Y.max() - self.Y.min() + 1


class NoisyMNIST(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        # 读取MATLAB文件
        mat_data = scipy.io.loadmat(path + 'NoisyMNIST.mat')
        # 输出MATLAB文件中的键（变量名）
        self.Y = mat_data['trainLabel'].astype(np.int32).reshape(50000, )
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        X1 = mat_data['X1'].astype(np.float32).reshape(-1, 784)
        X2 = mat_data['X2'].astype(np.float32).reshape(-1, 784)
        self.X = [X1,X2]
        self.num_views = 2
        self.num_classes = self.Y.max() - self.Y.min() + 1


class Hdigit(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        X1 = data['data'][0][0].T.astype(np.float32)
        X2 = data['data'][0][1].T.astype(np.float32)
        self.X = [X1,X2]
        self.num_views = 2
        self.num_classes = self.Y.max() - self.Y.min() + 1


class Cifar10(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        X1 = data['data'][0][0].T.astype(np.float32)
        X2 = data['data'][1][0].T.astype(np.float32)
        X3 = data['data'][2][0].T.astype(np.float32)
        self.X = [X1,X2,X3]
        self.num_views = 3
        self.num_classes = self.Y.max() - self.Y.min() + 1


class Prokaryotic(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551, )
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        X1 = data['X'][0][0].astype(np.float32)
        X2 = data['X'][1][0].astype(np.float32)
        X3 = data['X'][2][0].astype(np.float32)
        self.X = [X1,X2,X3]
        self.num_views = 3
        self.num_classes = self.Y.max() - self.Y.min() + 1


class PIE(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        data = scipy.io.loadmat(path + 'PIE_face_10.mat')
        scaler = MinMaxScaler((0, 1))
        self.Y = data['gt'].astype(np.int32)
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        data_X = data['X'][0]
        for v in range(len(data_X)):
            data_X[v] = data_X[v].T
        X1 = scaler.fit_transform(data_X[0].astype(np.float32))
        X2 = scaler.fit_transform(data_X[1].astype(np.float32))
        X3 = scaler.fit_transform(data_X[2].astype(np.float32))
        self.X = [X1,X2,X3]
        self.num_views = 3
        self.num_classes = self.Y.max() - self.Y.min() + 1


class Scene15(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        data = scipy.io.loadmat(path + 'scene15_mtv.mat')
        scaler = MinMaxScaler((0, 1))
        self.Y = data['gt'].astype(np.int32)
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        data_X = data['X'][0]
        for v in range(len(data_X)):
            data_X[v] = data_X[v].T
        X1 = scaler.fit_transform(data_X[0].astype(np.float32))
        X2 = scaler.fit_transform(data_X[1].astype(np.float32))
        X3 = scaler.fit_transform(data_X[2].astype(np.float32))
        self.X = [X1,X2,X3]
        self.num_views = 3
        self.num_classes = self.Y.max() - self.Y.min() + 1


# class Cora(BaseMultiViewDataset):
#     def __init__(self, path):
#         super().__init__()
#         data = h5py.File(path + 'Cora.mat', 'r')
#         scaler = MinMaxScaler((0, 1))
#         self.Y = np.asarray(data['Y'][()]).squeeze().astype(np.int32)
#         if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
#         if self.Y.min() == 1: self.Y = self.Y - 1
#         refs = np.asarray(data['X'][()]).reshape(-1)
#         data_X = data['X'][0]
#         for v in range(len(data_X)):
#             data_X[v] = np.asarray(data[refs[v]][()], dtype=np.float32)
#             data_X[v] = np.squeeze(data_X[v])
#         X1 = scaler.fit_transform(data_X[0].astype(np.float32))
#         X2 = scaler.fit_transform(data_X[1].astype(np.float32))
#         X3 = scaler.fit_transform(data_X[2].astype(np.float32))
#         self.X = [X1,X2,X3]
#         self.num_views = 3
#         self.num_classes = self.Y.max() - self.Y.min() + 1

class Cora(BaseMultiViewDataset):
    def __init__(self, path):
        super().__init__()
        data = h5py.File(path + 'Cora.mat', 'r')

        self.Y = np.asarray(data['Y'][()]).squeeze().astype(np.int32)
        if self.Y.ndim > 1:
            self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1:
            self.Y = self.Y - 1

        n_samples = len(self.Y)
        refs = np.asarray(data['X'][()]).reshape(-1)

        self.X = []
        for v in range(3):
            x = np.asarray(data[refs[v]][()], dtype=np.float32)
            x = np.squeeze(x)
            if x.shape[0] != n_samples and x.shape[1] == n_samples:
                x = x.T
            x = MinMaxScaler((0, 1)).fit_transform(x).astype(np.float32)
            self.X.append(x)

        self.num_views = 3
        self.num_classes = self.Y.max() - self.Y.min() + 1

        data.close()

class Unified_mv(BaseMultiViewDataset):
    def __init__(self, path, dataset):
        super().__init__()
        try:
            data = scipy.io.loadmat(path + dataset + '.mat')
        except Exception as e:
            print(f"scipy.io.loadmat failed: {e}")
            print("Trying h5py...")
            data = h5py.File(path + dataset + '.mat', 'r')

        data_X = data['X']
        self.num_views = data_X.shape[1]
        for i in range(self.num_views):
            data_X[0][i] = normalize(data_X[0][i])
            if sp.issparse(data_X[0][i]):
                data_X[0][i] = data_X[0][i].toarray()
            data_X[0][i] = np.asarray(data_X[0][i], dtype=np.float32)
        try:
            self.Y = data['truth'].flatten().astype(np.int32)
        except:
            self.Y = data['Y'].flatten().astype(np.int32)
        if self.Y.ndim > 1: self.Y = self.Y.reshape(-1)
        if self.Y.min() == 1: self.Y = self.Y - 1
        self.X = [data_X[0][i] for i in range(self.num_views)]
        self.num_classes = self.Y.max() - self.Y.min() + 1




def datasets(args):
    dataset = args.dataset
    data_path = './datasets/'

    if dataset == "BDGP":
        dataset = BDGP(data_path)
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST(data_path)
    elif dataset == "Fashion":
        dataset = Fashion(data_path)
    elif dataset == "HW":
        dataset = HW(data_path)
    elif dataset == "hw6":
        dataset = hw6(data_path)
    elif dataset == "NGs":
        dataset = NGs(data_path)
    elif dataset == "synthetic3d":
        dataset = synthetic3d(data_path)
    elif dataset == "Hdigit":
        dataset = Hdigit(data_path)
    elif dataset == "cifar10":
        dataset = Cifar10(data_path)
    elif dataset == "prokaryotic":
        dataset = Prokaryotic(data_path)
    elif dataset == "Caltech-2V":
        dataset = Caltech(data_path, view=2)
    elif dataset == "Caltech-3V":
        dataset = Caltech(data_path, view=3)
    elif dataset == "Caltech-4V":
        dataset = Caltech(data_path, view=4)
    elif dataset == "Caltech-5V":
        dataset = Caltech(data_path, view=5)
    elif dataset == "PIE":
        dataset = PIE(data_path)
    elif dataset == "Scene15":
        dataset = Scene15(data_path)
    elif dataset == "Cora":
        dataset = Cora(data_path)
    elif dataset in ['BBCnews', 'BBCsports', 'NGs', 'Citeseer', '3sources', 'MSRC-v1', 'ALOI', 'animals',
                     'Out_Scene', '100leaves', 'HW', 'MNIST', 'GRAZ02', 'Youtube', 'MNIST10k',
                     'Reuters', 'Wikipedia', 'Caltech101-7', 'Caltech102', 'ORL', 'NUS-WIDE',
                     'NoisyMNIST_15000', 'iaprtc12', 'YaleB_Extended']:
        dataset = Unified_mv(data_path, dataset)
    else:
        raise NotImplementedError
    return dataset


def load_data(args):
    dataset = datasets(args)

    dims = dataset.view_dim()
    view = dataset.num_views
    data_size = len(dataset)
    class_num = dataset.num_classes

    index = np.arange(data_size)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * data_size)], index[int(0.8 * data_size):]

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    if args.add_conflict or args.add_Noise:
        dataset.postprocessing(test_index = test_index, addNoise=args.add_Noise, sigma=0.5, ratio_noise=0.1, addConflict=args.add_conflict,
                               ratio_conflict=0.4)


    return dataset, dims, view, data_size, class_num, train_loader, test_loader



@dataclass
class MVDataBundle:
    features: List[torch.Tensor]
    labels: torch.Tensor
    dims: List[int]
    num_views: int
    num_classes: int
    idx_labeled: torch.Tensor
    idx_val: torch.Tensor
    idx_test: torch.Tensor
    idx_vt: torch.Tensor
    dataset_obj: object



def count_each_class_num(labels):
    count_dict = {}
    for label in labels:
        label = int(label)
        count_dict[label] = count_dict.get(label, 0) + 1
    return count_dict



def generate_partition(labels, ratio, seed):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)

    p_labeled = []
    p_unlabeled = []
    index = [i for i in range(len(labels))]
    if seed >= 0:
        rng = np.random.RandomState(seed)
        rng.shuffle(index)
    shuffled_labels = labels[index]
    for idx, label in enumerate(shuffled_labels):
        label = int(label)
        if labeled_each_class_num[label] > 0:
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
        else:
            p_unlabeled.append(index[idx])
    return p_labeled, p_unlabeled



def split_unlabeled(indices, val_ratio, seed):
    indices = list(indices)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    val_size = int(round(len(indices) * val_ratio))
    val_size = min(max(val_size, 1), max(len(indices) - 1, 1))
    idx_val = indices[:val_size]
    idx_test = indices[val_size:]
    if len(idx_test) == 0:
        idx_test = idx_val
    return idx_val, idx_test



def load_mv_data_bundle(args) -> MVDataBundle:
    dataset = datasets(args)
    dims = dataset.view_dim()
    num_views = dataset.num_views
    num_classes = dataset.num_classes
    labels_np = np.asarray(dataset.Y).astype(np.int64).reshape(-1)

    idx_labeled, idx_unlabeled = generate_partition(labels_np, ratio=args.ratio, seed=2026)
    idx_val, idx_test = split_unlabeled(idx_unlabeled, val_ratio=args.val_ratio, seed=2026 + 17)

    if args.add_conflict or args.add_Noise:
        dataset.postprocessing(test_index = idx_unlabeled, addNoise=args.add_Noise, sigma=0.5, ratio_noise=0.1, addConflict=args.add_conflict,
                               ratio_conflict=0.4)

    features = [torch.from_numpy(np.asarray(x)).float().to(args.device) for x in dataset.X]
    labels = torch.from_numpy(labels_np).long().to(args.device)

    return MVDataBundle(
        features=features,
        labels=labels,
        dims=list(dims),
        num_views=int(num_views),
        num_classes=int(num_classes),
        idx_labeled=torch.tensor(idx_labeled, dtype=torch.long, device=args.device),
        idx_val=torch.tensor(idx_val, dtype=torch.long, device=args.device),
        idx_test=torch.tensor(idx_test, dtype=torch.long, device=args.device),
        idx_vt=torch.tensor(idx_unlabeled, dtype=torch.long, device=args.device),
        dataset_obj=dataset,
    )