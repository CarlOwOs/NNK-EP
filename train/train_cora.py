import torch
import torch_geometric

from torch import nn
from torch_geometric import utils
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

import graph_model
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

            output = output[data.train_mask] # Node task
            target = target[data.train_mask] # Node task

            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), data.num_graphs)
            pred = output.argmax(dim=1)
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

            output = output[data.val_mask] # Node task
            target = target[data.val_mask] # Node task

            loss = loss_criterion(output, target)

            validation_loss.update(loss.item(), data.size(0))
            pred = output.argmax(dim=1)
            validation_accuracy.update(utils.accuracy(pred, target), count = len(target))
            tqdm_loader.set_postfix(Loss=validation_loss.avg, Accuracy=validation_accuracy.avg)

    return validation_loss.avg, validation_accuracy.avg


if __name__ == '__main__':

    hparams = {
        "batch_size":1, #64
        "num_epochs":40, #200
        "lr":1e-2,
    }
    wandb.init(project="NNK prunning (cora)", entity="carlowos", config = hparams)
    hparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Planetoid(root="./data/gcn/cora", name="cora")
    dataset = dataset.shuffle()

    train_loader = DataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        shuffle=True)

    validation_loader = DataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        shuffle=True)

    model = graph_model.XXSGCN(dataset.num_features, dataset.num_classes)

    model = model.to(hparams["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    loss_criterion = nn.CrossEntropyLoss()

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