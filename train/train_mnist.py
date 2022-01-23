import torch
import torch_geometric

from torch import nn
from torch_geometric import utils
from torch_geometric.datasets import GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

import graph_model
import graph_model_XS
import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import wandb

def train(model, train_loader, optimizer, loss_criterion, hparams, epoch):
    model.train()
    train_accuracy = metrics.AverageMeter()
    train_loss = metrics.AverageMeter()

    with tqdm(train_loader) as tqdm_loader:
        for batch_idx, data in enumerate(tqdm_loader):
            desc = '[TRAIN] Epoch:{}  Iter:[{}/{}]'\
                .format(epoch, batch_idx+1, len(train_loader))
            tqdm_loader.set_description(desc)

            data = data.to(hparams["device"])
            optimizer.zero_grad()
            output = model(data)
            target = data.y.to(hparams["device"])
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            pred = output.max(1)[1]
            train_accuracy.update(utils.accuracy(pred, target), count = len(target))
            tqdm_loader.set_postfix(Loss=train_loss.avg, Accuracy=train_accuracy.avg)

    return train_loss.avg, train_accuracy.avg

def val(model, validation_loader, loss_criterion, hparams):
    model.eval()
    validation_accuracy = metrics.AverageMeter()
    validation_loss = metrics.AverageMeter()

    with tqdm(validation_loader) as tqdm_loader:
        for batch_idx, data in enumerate(tqdm_loader):
            desc = '[VALIDATION] Iter:[{}/{}]'\
                .format(batch_idx+1, len(validation_loader))
            tqdm_loader.set_description(desc)

            data = data.to(hparams["device"])
            output = model(data)
            target = data.y.to(hparams["device"])
            loss = loss_criterion(output, target)

            validation_loss.update(loss.item(), data.size(0))
            pred = output.max(1)[1]
            validation_accuracy.update(utils.accuracy(pred, target), count = len(target))
            tqdm_loader.set_postfix(Loss=validation_loss.avg, Accuracy=validation_accuracy.avg)

    return validation_loss.avg, validation_accuracy.avg


if __name__ == '__main__':
    
    hparams = {
        "batch_size":16, #64
        "num_epochs":40, #200
        "lr":1e-4,
    }
    wandb.init(project="NNK prunning", entity="carlowos", config = hparams)
    hparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    loss_criterion = nn.CrossEntropyLoss()

    model = graph_model.Small5GCN(1, 2)

    model = model.to(hparams["device"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    train_dataset = GNNBenchmarkDataset(root="./data/gcn/MNIST_norm", name="MNIST",
        split="train", pre_transform=T.Compose([T.NormalizeScale(), T.NormalizeFeatures()]))
    
    # Select only classes 0 and 1
    idx = [i for i, graph in enumerate(train_dataset) if graph.y < 2]
    train_dataset = train_dataset[idx]

    validation_dataset = GNNBenchmarkDataset(root="./data/gcn/MNIST_norm", name="MNIST",
        split="val", pre_transform=T.Compose([T.NormalizeScale(), T.NormalizeFeatures()]))
    
    # Select only classes 0 and 1
    idx = [i for i, graph in enumerate(validation_dataset) if graph.y < 2]
    validation_dataset = validation_dataset[idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True)

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True)

    tr_losses = []
    tr_accuracies  = []
    val_losses = []
    val_accuracies = [] 

    for epoch in range(1, hparams["num_epochs"] + 1):

        train_loss, tr_accuracy = train(model, train_loader, optimizer, loss_criterion, hparams, epoch)
        tr_losses.append(train_loss)
        tr_accuracies.append(tr_accuracy)
        val_loss, val_accuracy = val(model, validation_loader, loss_criterion, hparams)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        wandb.log({"train_loss": train_loss, "train_accuracy": tr_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})

    wandb.watch(model)