
import argparse
import os
import pickle
import random
import sys
import time
from importlib import import_module

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import ArgoDataset as Dataset, from_numpy, ref_copy, collate_fn
from utils import Logger, load_pretrain, gpu
# torch.cuda.set_device(3)
os.umask(0)


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(
    description="Data preprocess for argo forcasting dataset"
)
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)


def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, *_ = model.get_model()

    config["preprocess"] = False # we use raw data to generate preprocess data
    config["val_workers"] = 4
    config["workers"] = 4
    config['cross_dist'] = 6
    config['cross_angle'] = 0.5 * np.pi

    os.makedirs(os.path.dirname(config["train_split"]),exist_ok=True)    

    # val(config)
    # test(config)
    train(config)
    # import ipdb;ipdb.set_trace()


def train(config):
    # Data loader for training set
    dataset = Dataset("dataset/train_mini/data", config, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    stores = [None for x in range(110)]
    t = time.time()
    for i, data in enumerate(tqdm(train_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
                'trajs2',
                'traj1'
            ]:                 # 10 dimension
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()



    dataset = PreprocessDataset(stores, config, train=True)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader,config["preprocess_train"])


def val(config):
    # Data loader for validation set
    dataset = Dataset("dataset/val_mini/data", config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    # stores = [None for x in range(39472)]
    stores = [None for x in range(480)]

    t = time.time()
    for i, data in enumerate(tqdm(val_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
                'trajs2',
                'traj1',
                'argo_id'
            ]:          # 10 dimension
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)

    modify(config, data_loader,config["preprocess_val_mini_noise"])


def test(config):
    dataset = Dataset(config["test_split"], config, train=False)
    test_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )


    stores = [None for x in range(78143)]

    t = time.time()
    for i, data in enumerate(tqdm(test_loader)):
        data = dict(data)
        # data.keys() = dict_keys(['city', 'trajs', 'steps', 'feats', 'ctrs', 'orig', 'theta', 'rot', 'gt_preds', 'has_preds', 'idx', 'graph'])
        # import ipdb;ipdb.set_trace()

        for j in range(len(data["idx"])):
            # data['idx'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            store = dict()
            # import ipdb;ipdb.set_trace()

            for key in [
                "idx",      # 0
                "city",     # PIT
                "feats",    # store['feats'].shape = (26, 20, 3)
                "ctrs",     # store['ctrs'].shape = (26, 2)
                "orig",     # store['orig'] = array([1536.718  ,  241.83821], dtype=float32)
                "theta",    # store['theta'] = 2.5896114733142612
                "rot",      # store['rot'] = array([[-0.85148734, -0.5243752 ],
                            #                         [ 0.5243752 , -0.85148734]], dtype=float32)
                "graph",    # dict_keys(['ctrs', 'num_nodes', 'feats', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs'])
                            # store['graph']['ctrs'].shape = (1674, 2), num_nodes = 1674, feats.shape = (1674, 2), 
            ]:              # 8 dimention
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store
            # import ipdb;ipdb.set_trace()

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config, train=False)
    # print("dataset:", dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['workers'],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=True,
        drop_last=False)
    
    # global data_all
    data_all = next(iter(data_loader))
    
    # data_all[0].keys() ---   dict_keys(['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats', 'idx'])

    # data_all[0]['lane_idcs'] = tensor([  0,   0,   0,  ..., 185, 185, 185], dtype=torch.int16)
    # data_all[0]['lane_idcs'].size() = torch.Size([1674])

    # data_all[0]['ctrs'] = 
    # tensor([[ 30.6144,  -7.2681],
    # [ 33.3912,  -7.8745],
    # [ 36.1679,  -8.4810],
    # ...,
    # [-28.1613,  56.5973],
    # [-27.8702,  56.7838],
    # [-27.5790,  56.9703]])
    # data_all[0]['ctrs'].size() = torch.Size([1674, 2])

    # data_all[0]['pre_pairs'] =
    # tensor([[  0,  79],
    #         [  1,  40],
    #         [  2,   5],
    #         [  2,  18],
    #         [  3, 131],
    #         [  4,   8],
    #         [  5,  66],
    #         [  7,  67],
    #         [  9,   8],
    #         [ 10, 131],
    #         [ 11, 109],
    #         [ 12,  99],
    #         [ 13, 127],
    #         [ 14,  65],    torch.Size([192, 2])

    # data_all[0]['pre_pairs'].size = torch.Size([192, 2])   
    # data_all[0]['suc_pairs'].size() = torch.Size([192, 2])
    # data_all[0]['left_pairs'].size() = torch.Size([115, 2])
    # data_all[0]['right_pairs'].size() = torch.Size([65, 2])


    # data_all[0]['feats'].size() = torch.Size([1674, 2])

    # data_all[0]['idx'] = 0

    modify(config, data_loader,config["preprocess_test"])
    # import ipdb;ipdb.set_trace()

def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data


def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data



def modify(config, data_loader, save):
    t = time.time()
    store = data_loader.dataset.split
    for i, data in enumerate(data_loader):
        data = [dict(x) for x in data]

        out = []
        for j in range(len(data)):
            out.append(preprocess(to_long(gpu(data[j])), config['cross_dist']))

        for j, graph in enumerate(out):
            idx = graph['idx']
            store[idx]['graph']['left'] = graph['left']
            store[idx]['graph']['right'] = graph['right']

        if (i + 1) % 100 == 0:
            print((i + 1) * config['batch_size'], time.time() - t)
            t = time.time()

    f = open(os.path.join(root_path, 'preprocess', save), 'wb')
    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

class PreprocessDataset():
    def __init__(self, split, config, train=True):
        self.split = split
        self.config = config
        self.train = train

    def __getitem__(self, idx):
        from data import from_numpy, ref_copy

        data = self.split[idx]
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
            # print(data['graph'].keys())
            # print(key)
            # print('fffffffffffffffffffffffffffffffffffffffffffffffffffffffffyy')
            graph[key] = ref_copy(data['graph'][key])
        graph['idx'] = idx
        return graph

    def __len__(self):
        return len(self.split)




def preprocess(graph, cross_dist, cross_angle=None):
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    if cross_angle is not None:
        f1 = graph['feats'][hi]
        f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < config['cross_angle'])
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -config['cross_angle'])
        right_mask = mask.logical_not()

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    out['idx'] = graph['idx']
    return out



def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


if __name__ == "__main__":
    main()