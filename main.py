import argparse
from os import path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv, GATConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorflow.python.client import device_lib
import time
from model import GATModel
from train_ppi_baseline import train, test, collate_fn

MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "gat_model_state.pth")

def main(args):
    # create the dataset
    train_dataset, valid_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="valid"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]
    print("Number of features: ", n_features, " Number of classes: ", n_classes)

    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))
    model = GATModel(graph=train_dataset.graph, n_heads = 8, n_layers=3, input_size=n_features,
                            hidden_size=256, output_size=n_classes, nonlinearity = F.leaky_relu).to(device)
    #model = MyGAT(graph=train_dataset.graph, n_layers=3, input_size=n_features, output_size=n_classes).to(device)
    
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.5)

    # train and test
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, scheduler, train_dataloader, valid_dataset, epochs = args.epochs)
        torch.save(model.state_dict(), MODEL_STATE_FILE)
    model.load_state_dict(torch.load(MODEL_STATE_FILE))
    return test(model, loss_fcn, device, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    main(args)

