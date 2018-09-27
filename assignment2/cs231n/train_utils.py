import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np
import queue
from numpy.random import randint

import matplotlib.pyplot as plt

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def accuracy(loader, model, its=None):
    model.eval() # set each model to evaluation mode
    num_correct = 0
    num_samples = 0
    preds = None
    with torch.no_grad():
        for (t, (x, y)) in enumerate(loader):
            break_its = its is not None and t > its
            x = x.to(device=device, dtype=dtype)  # move
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, p = scores.max(1)
            if preds is None:
                preds = p
            else:
                preds = torch.cat((preds, p))
            num_correct += (p == y).sum()
            num_samples += p.size(0)
            if break_its: break
    return preds, 100 * float(num_correct) / num_samples

def train_batch(x, y, model, optimizer):
    model.train()  # put model to training mode
    x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
    y = y.to(device=device, dtype=torch.long)
    scores = model(x)
    loss = F.cross_entropy(scores, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train(model, optimizer, loader_train, loader_val, epochs=1, its=None,
          eval_its=None, log_every=100, verbose=False):
    history = History()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            break_its = its is not None and t > its
            loss = train_batch(x, y, model, optimizer)
            if t % log_every == 0 or break_its:
                _, train_acc = accuracy(loader_train, model, its=eval_its)
                preds, val_acc = accuracy(loader_val, model, its=eval_its)
                history.update(loss.item(), train_acc, val_acc, preds)
                if verbose:
                    print('It %d, loss = %.4f' % (t, loss.item()))
                    print('Train acc \t= %.2f' % history.train_acc)
                    print('Val acc \t= %.2f\n' % history.val_acc)
            if break_its: break
    return history

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def construct_model(parameters, im_shape, num_classes):

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
            # TODO: stride and new im_size
            layers.append(nn.Conv2d(channels, filter_count, filter_size, padding=pad))
            #layers.append(nn.Dropout2d(parameters["Dropout"]))
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

class History(object):
    def __init__(self):
        self.losses = []
        self.train_accs = []
        self.val_accs = []
        self.preds = []
    def update(self, loss, train_acc, val_acc, preds):
        self.losses.append(loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        if self.preds:
            assert(preds.shape == self.preds[-1].shape)
        self.preds.append(preds)
    @property
    def train_acc(self):
        return self.train_accs[-1] if self.train_accs else None
    @property
    def val_acc(self):
        return self.val_accs[-1] if self.val_accs else None
    @property
    def bad(self):
        return 0
        return (self.val_accs[-1] - self.val_accs[0]) < 5.
    @property
    def working(self):
        ln = self.losses[-1]
        l0 = self.losses[0]
        perc_dec = 100. * (l0 - ln) / l0
        return perc_dec > 10.

class ChoiceSampler(object):
    def __init__(self, choices):
        self.choices = choices
    def sample(self):
        ind = range(len(self.choices))
        return self.choices[ind]

class HyperOpt(object):
    """
    A hyperparameter optimizer. Includes individual parameter choices,
    training routines, and coarse/fine strategies.
    
    Example usage:

    choices = {
      "FilterSize": [3, 5],
      "FilterCount": [8, 32, 128],
    }
    ho = HyperOpt(choices, construct_model, train, loader_train, loader_val)
    ho.optimize()
    """
    def __init__(self, choices, constructor, trainer, loader_train, loader_val,
                 max_active=None, coarse_its=None, fine_epochs=10, verbose=False,
                 coarse_fine="contract"):
        """
        Construct a HyperOpt object.
        
        Required arguments:
        - choices: Dictionary from parameter name to parameter choices.
        - constructor: Model constructor function. Must have the following API:
          - constructor(parameters, image_shape)
        - trainer: Model training function. Must have the following API:
          - trainer(model, loader_train, loader_val)
        - trainer_val: PyTorch DataLoader for training set.
        - loader_val: PyTorch DataLoader for validation set.
        - max_active: Maximum number of active parameter sets.
        - coarse_its: Number of training iterations during coarse phases.
        - fine_epochs: Number of training epochs during fine phases.
        - coarse_fine: Coarse/fine strategy. One of the following:
          - contracting: Generate max_active and then complete or kill all,
            before repeating.
          - filling: Always keep active at max_active.
        """
        self.choices = choices
        self.num_choices = np.prod(list(map(len, choices)))
        self.constructor = constructor
        self.trainer = trainer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.max_active = max_active
        self.coarse_its = coarse_its
        self.fine_epochs = fine_epochs
        self.coarse_fine = coarse_fine
        self.verbose = verbose
        self.fine = False # begin coarse
        self.sampled = set([])
        self.active = queue.PriorityQueue() # begin with no active
        self.trained = queue.PriorityQueue() # begin with no trained
        self.results = [] # begin with no results

    def optimize(self):
        """
        Optimize hyperparameters.
        """
        fine = False
        while len(self.sampled) < num_choices or self.active:
            # If the coarse search is done, move to fine.
            if not fine and self.active.qsize() == self.max_active:
                fine = True
            # If the fine search is done, move to coarse.
            elif fine and not self.active:
                fine = False
            # Step.
            self.coarse_step() if fine else self.fine_step()

    def coarse_step(self):

        # Sample parameters and check if this was sampled previously.
        lrs, parameter_inds, parameter_vals = [], {}, {}
        for (name, vals) in self.choices.items():
            ind = randint(0, len(self.choices))
            parameter_inds[name] = ind
            parameter_vals[name] = self.choices[name][ind]
        sample = tuple(sorted(parameter_inds.items()))
        if sample in self.sampled:
            return
        sampled.add(sample)
        
        # Find the right learning rate (if there is one).
        lrs = []
        while len(lrs) < 4:
            lr = 10 ** np.random.randint(-6, 1)
            if lr in lrs: continue
            model = self.constructor(parameters, (3,32,32), 10) # TODO: don't hard code?
            print(model)
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=0.9, nesterov=True)
            history = self.trainer(model, optimizer, self.loader_train,
                                   self.loader_val, epochs=1,
                                   its=self.coarse_its, eval_its=16,
                                   verbose=True)#self.verbose)
            # If it's working, add it to our active set and break.
            if history.working:
                print("\tlr = %.2E" % lr)
                print("\t\tworking = %s" % history.working)
                self.active.put((-history.val_acc, model, optimizer, history))
                break
            
    def fine_step(self):
        _, model, optimizer, history = self.active.get()
        history = self.trainer(model, optimizer, self.loader_train,
                               self.loader_val, epochs=self.fine_epochs,
                               eval_its=16)
        self.trained.put((-history.val_acc, model, history))

    def step(self):

        """
        plt.subplot(2, 1, 1)
        #plt.plot(history.losses, 'o')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        
        plt.subplot(2, 1, 2)
        #plt.plot(history.train_acc, '-o')
        #plt.plot(history.val_acc, '-o')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
        """
        pass

    @property
    def best_val_acc(self):
        if self.trained.qsize():
            return -self.trained.queue[0][0]
        else:
            0.0
