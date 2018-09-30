import inspect
import logging
import numpy as np
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOSSES = "losses.txt"
TRAIN_ACCS = "train_accs.txt"
VAL_ACCS = "val_accs.txt"

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def construct_model(parameters, im_shape, num_classes):
    """
    Construct a model given parameters and image/class info.
    
    Arguments:
    - parameters: Parameter dictionary from name to value.
    - im_shape: 3-tuple of (channels, rows, cols).
    - num_classes: Number of classes.
    """

    (channels, im_size, im_size2) = im_shape
    assert(im_size == im_size2)

    layers = []

    # For each conv pattern.
    N = parameters["N"]
    for conv_layer in range(N):
        # Filter size and filter count.
        filter_size = parameters["FilterSize"]
        filter_count = parameters["FilterCount"]
        # Architecture
        if parameters["Architecture"] == "conv-relu-pool":
            continue
        elif parameters["Architecture"] == "batchnorm-relu-conv":
            layers.append(torch.nn.BatchNorm2d(channels))
            layers.append(torch.nn.ReLU())
            pad = filter_size // 2
            stride = parameters["Stride"] if im_size // parameters["Stride"] >= 4 else 1
            im_size //= stride
            layers.append(nn.Conv2d(channels, filter_count, filter_size, stride=stride, padding=pad))
            layers.append(nn.Dropout2d(parameters["Dropout"]))
        channels = filter_count

    # For each fc pattern.
    layers.append(Flatten())
    in_dim = channels * im_size * im_size
    M = parameters["M"]
    for fc_layer in range(M):
        out_dim = parameters["HiddenSize"] if fc_layer < M-1 else num_classes
        layers.append(nn.Linear(in_dim, out_dim))
        #layers.append(nn.Dropout(parameters["Dropout"]))
        in_dim = out_dim

    return nn.Sequential(*layers)

def accuracy(loader, model, device, iterations=None):
    """
    Evaluate the model on the data in loader.
    
    Arguments:
    - loader: Data loader.
    - model: PyTorch model.
    - device: Device.
    - iterations: Iterations.
    """
    model.eval() # set each model to evaluation mode
    num_correct = 0
    num_samples = 0
    preds = None
    if iterations is None:
        iterations = len(loader_train)
    with torch.no_grad():
        for (it, (x, y)) in enumerate(loader):
            x = x.to(device=device, dtype=torch.float32)  # move
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, p = scores.max(1)
            if preds is None:
                preds = p
            else:
                preds = torch.cat((preds, p))
            num_correct += (p == y).sum()
            num_samples += p.size(0)
            if it > iterations:
                break
    return preds, 100 * float(num_correct) / num_samples

def train_batch(x, y, model, optimizer, device):
    """
    Train model on the given (x,y) batch given the optimizer.
    
    Arguments:
    - x: Data.
    - y: Labels.
    - optimizer: PyTorch optimizer.
    - device: Device.
    """
    model.train()  # put model to training mode
    x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
    y = y.to(device=device, dtype=torch.long)
    scores = model(x)
    loss = F.cross_entropy(scores, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train(loader_train, loader_val, model, optimizer,
          logdir, log_every=100, verbose=False, vis=None,
          epochs=1, iterations=None, eval_iterations=None, device=None):
    """
    Train the model given the optimizer, data, and other settings.
    
    Arguments:
    - loader_train: Training data loader.
    - loader_val: Validation data loader.
    - model: PyTorch model.
    - optimizer: PyTorch optimizer.
    - logdir: Logging directory.
    - log_every: How many iterations pass before logging.
    - verbose: Whether to log to console.
    - vis: Visualizer.
    - epochs: How many training epochs.
    - iterations: How many traing iterations.
    - eval_iterations: How many validation iterations.
    - device: Device.
    - iterations: 
    """
    fname = inspect.stack()[0][3]
    logger.debug('%s: starting' % (fname))
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    if iterations is None:
        iterations = len(loader_train)
    losses = []
    for e in range(1, epochs+1):
        for it, (x, y) in enumerate(loader_train, start=1):
            # Compute, log, and visualize the loss.
            loss = train_batch(x, y, model, optimizer, device).item()
            losses.append(loss)
            open(join(logdir, LOSSES), "a").write("%f\n"%loss)
            # Evaluate if it's a logging iteration or the end.
            if it==1 or it % log_every == 0 or it == iterations:
                _, train_acc = accuracy(loader_train, model, device, iterations=eval_iterations)
                open(join(logdir, TRAIN_ACCS), "a").write("%f\n"%train_acc)
                _, val_acc = accuracy(loader_val, model, device, iterations=eval_iterations)
                open(join(logdir, VAL_ACCS), "a").write("%f\n"%val_acc)
                if verbose:
                    logger.info('%s: Epoch %d/%d, It %d/%d, loss = %.4f' % (fname, e, epochs, it, iterations, loss))
                    logger.info('%s: Train acc = %.2f' % (fname, train_acc))
                    logger.info('%s: Val acc = %.2f\n' % (fname, val_acc))
                if vis: vis.update()
            if it == iterations: break
        # Save a checkpoint at the end of the epoch.
        ch = Checkpoint(model, optimizer, e, val_acc)
        ch_fname = join(logdir, "checkpoint-%02d-%03d.pt.tar" % (e, it))
        ch.save(ch_fname)
    vis.update(table=True)
    logger.debug('%s: ending' % (fname))
    return losses, ch

class Checkpoint(object):
    def __init__(self, model=None, optimizer=None, epoch=None, acc=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.acc = acc
    def save(self, filename):
        fname = inspect.stack()[0][3]
        logger.debug('%s' % fname)
        r = {
            "epoch": self.epoch,
            "model": self.model,
            "optimizer": self.optimizer,
            "acc": self.acc
            }
        torch.save(r, filename)
    def load(self, filename):
        fname = inspect.stack()[0][3]
        logger.debug('%s' % fname)
        r = torch.load(filename)
        self.model = r["model"]
        self.optimizer = r["optimizer"]
        self.epoch = r["epoch"]
        self.acc = r["acc"]
