import inspect
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def accuracy(loader, model, device, its=None):
    """
    Evaluate the model on the data in loader for a maximum of its
    iterations (batches).
    """
    model.eval() # set each model to evaluation mode
    num_correct = 0
    num_samples = 0
    preds = None
    with torch.no_grad():
        for (t, (x, y)) in enumerate(loader):
            break_its = its is not None and t > its
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
            if break_its: break
    return preds, 100 * float(num_correct) / num_samples

def train_batch(x, y, model, optimizer, device):
    """
    Train model on the given (x,y) batch given the optimizer.
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

def train(model, optimizer, loader_train, loader_val, epochs=1, its=None,
          eval_its=None, log_every=100, verbose=False, device=None, history=None, vis=None):
    """
    Train the model for the given number of epochs and iterations on the data in
    the loader_train.
    
    Evaluate every log_every on the data in loader_val.
    
    Update and return training the training history.
    """
    fname = inspect.stack()[0][3]
    logger.debug('%s: starting' % (fname))
    if history is None:
        history = History()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    if its is None: its = len(loader_train)
    for e in range(1, epochs+1):
        for t, (x, y) in enumerate(loader_train, start=1):
            loss = train_batch(x, y, model, optimizer, device)
            if t==1 or t % log_every == 0 or t == its:
                _, train_acc = accuracy(loader_train, model, device, its=eval_its)
                preds, val_acc = accuracy(loader_val, model, device, its=eval_its)
                history.update(loss.item(), train_acc, val_acc, preds, vis)
                if verbose:
                    logger.info('%s: Epoch %d/%d, It %d/%d, loss = %.4f' % (fname, e, epochs, t, its, loss.item()))
                    logger.info('%s: Train acc = %.2f' % (fname, history.train_acc))
                    logger.info('%s: Val acc = %.2f\n' % (fname, history.val_acc))
            if t == its: break
    logger.debug('%s: ending' % (fname))
    return history

def construct_model(parameters, im_shape, num_classes):
    fname = inspect.stack()[0][3]
    
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

class History(object):
    num = 0
    def __init__(self):
        History.num += 1 # not threadsafe, but yolo
        self.id = History.num
        self.losses = []
        self.train_accs = []
        self.val_accs = []
        self.preds = []
        self.loss_window = None
    def update(self, loss, train_acc, val_acc, preds, vis):
        # Update data.
        self.losses.append(loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        if self.preds:
            assert(preds.shape == self.preds[-1].shape)
        self.preds.append(preds)
        # Visualize.
        if vis and not self.loss_window:
            opts = {
                "title": "Model %d Loss" % self.id,
                "xlabel": "iteration",
                "ylabel": "loss",
                "xtickmin": 0,
                "xtickmax": 50,
                "ytickmin": 0,
                "ytickmax": 5
                }
            self.loss_window = vis.line(Y=self.losses, opts=opts)
            opts["title"] = "Model %d Accuracy" % self.id
            opts["ylabel"] = "accuracy"
            opts["ytickmax"] = 100
            opts["legend"] = ["Train", "Val"]
            self.acc_window = vis.line(Y=list(zip(self.train_accs, self.val_accs)), opts=opts)
        elif vis:
            x = len(self.losses)
            vis.line(X=[x], Y=[loss], win=self.loss_window, update='append')
            vis.line(X=[[x, x]], Y=[(train_acc, val_acc)], win=self.acc_window,
                     update='append', opts={"legend": ["Train", "Val"]})
    def __lt__(self, other):
        return self.val_acc > other.val_acc # sort in decreasing order
    @property
    def train_acc(self):
        return self.train_accs[-1] if self.train_accs else None
    @property
    def val_acc(self):
        return self.val_accs[-1] if self.val_accs else None
    @property
    def working(self):
        ln = self.losses[-1]
        l0 = self.losses[0]
        perc_dec = 100. * (l0 - ln) / l0
        print("perc_dec =", perc_dec)
        return perc_dec > 5. # working if improves by 5%
