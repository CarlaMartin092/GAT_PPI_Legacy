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

def train(model, loss_fcn, device, optimizer, scheduler, train_dataloader, test_dataset, epochs):
    for epoch in range(epochs):
        model.train()
        losses = []
        t1 = time.time()
        for batch, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            model.g = subgraph
            
            for i in range(model.n_layers):
                model.convs[i].g = subgraph
                model.linear[i].g = subgraph
                if (i < model.n_layers - 1): model.bns[i].g = subgraph
            logits = model(features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        t2 = time.time()
        print("Epoch {:05d} | Loss: {:.4f} | Time: {:.3f}".format(epoch + 1, loss_data, t2-t1))

        if (epoch+1) % 5 == 0:
            scores = []
            val_loss = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                features = features.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                score, loss = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
                val_loss.append(loss)

            print("F1-Score: {:.4f} | Validation Loss: {:.4f} ".format(np.array(scores).mean(), np.array(val_loss).mean()))
    scheduler.step()

def test(model, loss_fcn, device, test_dataloader):
    test_scores = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores

def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for i in range(model.n_layers):
            model.convs[i].g = subgraph
            model.linear[i].g = subgraph
            if (i < model.n_layers - 1): model.bns[i].g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()

def collate_fn(sample):
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels
