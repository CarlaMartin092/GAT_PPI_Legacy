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


MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "gat_model_state.pth")


class BasicGraphModel(nn.Module):

    def __init__(self, graph, n_heads, n_layers, input_size, hidden_size, output_size, nonlinearity, dropout = 0.6):
        super().__init__()

        self.n_layers = n_layers
        self.g = graph
        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * hidden_size if i > 0 else input_size
            out_hidden = hidden_size if i < n_layers - 1 else output_size
            out_channels = n_heads

            self.convs.append(GATConv(in_hidden, out_hidden, num_heads=n_heads, attn_drop=0))
            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))
        
        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = nonlinearity


    def forward(self, feat):
        h = feat
        h = self.dropout0(h)

        for i in range(self.n_layers):
            conv = self.convs[i](self.g, h)
            linear = self.linear[i](h).view(conv.shape)

            h = conv + linear

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h, negative_slope=0.2)
                h = self.dropout(h)

        h = h.mean(1)

        return h

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
    model = BasicGraphModel(graph=train_dataset.graph, n_heads = 8, n_layers=3, input_size=n_features,
                            hidden_size=256, output_size=n_classes, nonlinearity = F.leaky_relu).to(device)
    #model = MyGAT(graph=train_dataset.graph, n_layers=3, input_size=n_features, output_size=n_classes).to(device)
    
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.5)

    # train and test
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, scheduler, train_dataloader, valid_dataset)
        torch.save(model.state_dict(), MODEL_STATE_FILE)
    model.load_state_dict(torch.load(MODEL_STATE_FILE))
    return test(model, loss_fcn, device, test_dataloader)


def train(model, loss_fcn, device, optimizer, scheduler, train_dataloader, test_dataset):
    for epoch in range(args.epochs):
        model.train()
        losses = []
        t1 = time.time()
        for batch, data in enumerate(train_dataloader):
            #print("Batch n° {}".format(batch))
            subgraph, features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            model.g = subgraph
            #for layer in model.gat_layers:
                #layer.g = subgraph
            #for layer in model.conv_layers:
                #layer.g = subgraph
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
        #for layer in model.gat_layers:
            #layer.g = subgraph
        #for layer in model.conv_layers:
            #layer.g = subgraph
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    main(args)
