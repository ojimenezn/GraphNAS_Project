import numpy as np
from hyperopt import fmin, tpe, hp
import os
import torch
from baseline.arma.arma import args, Net, run
import os.path as osp
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
seed = 123
os.environ["HYPEROPT_FMIN_SEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def load_data(dataset="Cora", supervised=True, ):
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["CS", "Physics"]:
        dataset = Coauthor(path, dataset, transform=T.NormalizeFeatures())
    elif dataset in ["Computers", "Photo"]:
        dataset = Amazon(path,dataset, transform=T.NormalizeFeatures())
    elif dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    if supervised:

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.train_mask[:-1000] = 1
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.val_mask[-1000: -500] = 1
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.test_mask[-500:] = 1
    data.num_classes = data.y.max().item() + 1
    return dataset

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
    mask[index] = 1
    return mask
def label_splits(data, num_classes, shuffle=False):
    if shuffle:
        indices = torch.randperm(data.x.size(0))

        data.train_mask = index_to_mask(indices[:-1000], size=data.num_nodes)
        data.val_mask = index_to_mask(indices[-1000:-500], size=data.num_nodes)
        data.test_mask = index_to_mask(indices[-500:], size=data.num_nodes)

        return data
    else:
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.train_mask[:-1000] = 1
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.val_mask[-1000: -500] = 1
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.test_mask[-500:] = 1

        return data

lr_list = [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
decay_list = [0, 1e-3, 1e-4, 1e-5, 5e-5, 5e-4]
dim_list = [8, 16, 32, 64, 128, 256, 512]

space = {
    'learning_rate' : hp.choice("learning_rate",lr_list),
    'dropout' : hp.choice("dropout", dropout_list),
    'weight_decay' : hp.choice("weight_decay", decay_list),
    'hidden_unit' : hp.choice("hidden_unit", dim_list),
}


if __name__ == "__main__":
    def f(params):
        args.skip_dropout = args.dropout = params["dropout"]
        args.hidden = params["hidden_unit"]
        try:
            val_score, test_score = run(dataset, Net(dataset), 1, args.epochs, params["learning_rate"], params["weight_decay"],
        args.early_stopping, label_splits)
            print(val_score)

            return - val_score
        except Exception as e:
            print(0)
            return 0


    dataset_list = ["Computers", "Photo"] # "CS", "Physics", "Computers", "Photo", "Pubmed" , "Citeseer", "Cora"
    for dataset_name in dataset_list:
        dataset = load_data(dataset_name, supervised=True)
        best = fmin(
            fn=f,
            space=space,
            algo=tpe.suggest,
            max_evals=5,
            show_progressbar=False)
        print(dataset_name, best)
        learning_rate = lr_list[best["learning_rate"]]
        dropout = dropout_list[best["dropout"]]
        weight_decay = decay_list[best["weight_decay"]]
        hidden_unit = dim_list[best["hidden_unit"]]
        print(dataset, f"learning_rate:{learning_rate},dropout:{dropout},weight_decay:{weight_decay},hidden_unit:{hidden_unit}")
        results = []
        print("_"*80)
        args.skip_dropout = args.dropout = dropout
        args.hidden = hidden_unit
        print("dataset:", dataset_name)
        torch.cuda.empty_cache()
        run(dataset, Net(dataset), 100, args.epochs, learning_rate, weight_decay,
                       args.early_stopping, label_splits)
        print("_"*80)