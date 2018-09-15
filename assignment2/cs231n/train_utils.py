import torch
import torch.nn as nn

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

def train(model, optimizer, loader_train, loader_val, epochs=1, its=None, eval_its=None):
    history = History()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            break_its = its is not None and t > its
            loss = train_batch(x, y, model, optimizer)
            if t % print_every == 0 or break_its:
                _, train_acc = accuracy(loader_train, model, its=eval_its)
                preds, val_acc = accuracy(loader_val, model, its=eval_its)
                history.update(loss, train_acc, val_acc, preds)
                print('It %d, loss = %.4f' % (t, loss.item()))
                print('Train acc \t= %.2f' % history.train_acc)
                print('Val acc \t= %.2f\n' % history.val_acc)
            if break_its: break
    return history

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
        return (self.val_accs[-1] - self.val_acs[0]) < 5.

class ChoiceSampler(object):
    def __init__self(self, choices):
        self.bad = np.zeros(len(choices))
        self.tot = np.zeros(len(choices))
        self.choices = choices
    def sample(self):
        good = (self.bad / (self.tot+1)) < 0.75
        assert(np.sum(good))
        ind = np.choice(np.nonzero(good)[0], 1)
        return self.choices[ind]
    def update(self, bad, val):
        ind = [i for (i,v) in enumerate(self.choices) if abs(v-val) < 1e-9][0]
        self.bad[ind] += bad
        self.tot[ind] += 1

class SetSampler(object):
    """
    Samples sets of parameters according to a schedule and taking into account
    past performance.

    Example usage:
    
    samplers = {
      "FilterSize": ChoiceSampler([3, 5]),
      "FilterCount": ChoiceSampler([8, 32, 128]),
    }
    sampler = SetSampler(samplers, fine_epochs=5)
    while sampler.num_complete < 10: # until 10 sets have each run for 5 epochs
      parameters = sampler.sample_parameters()
      model = construct_model(parameters) # assume you write
      history = train(model, loader_train, loader_val) # assume you write
      sampler.update(parameters, history)    
    """
    def __init__(self, samplers, num_active=None, course_its=None, fine_epochs=10,
                 course_fine="contract"):
        """
        Construct a SetSampler object.
        
        Required arguments:
        - samplers: a dictionary from 
        Convenience layer that perorms an affine transform followed by a ReLU
        
        Inputs:
        - x: Input to the affine layer
        - w, b: Weights for the affine layer
        
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        self.samplers = samplers
        self.results = []
        self.active = []
        self.num_active = 
        self.course_its = course_its
        self.fine_epochs = fine_epochs
    def sample_parameters(self):
        while 
        return dict([(n, p.sample()) for (n, s) in self.samplers.items()])
    def update(parameters, hist):
        [self.samplers[n].update(hist.bad, v) for (n, v) in parameters.items()]


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
            
def construct_model(parameters, im_shape):

    (channels, im_size, im_size2) = im_shape
    assert(im_size == im_size2)
    
    layers = []
    
    # For each conv pattern.
    for conv_layer in range(N):
        # Filter size and filter count.
        filter_size = parameters["FilterSize"]
        filter_count = parameters["FilterCount"]
        # Architecture
        if parameters["Architecture"] == "conv-relu-pool":
            continue
        elif parameters["Architecture"] == "batchnorm-relu-conv":
            layers.append(torch.nn.BatchNorm2d(channels))
            layers.append(layers.nn.ReLU())
            pad = filter_size // 2
            # TODO: stride and new im_size
            layers.append(nn.Conv2d(channels, filter_count, filter_size, padding=pad))
            #layers.append(nn.Dropout2d(parameters["Dropout"]))
        channels = filter_count
    
    # For each fc pattern.
    layers.append(Flatten())
    in_dim = channels * im_size * im_size
    for fc_layer in range(M):
        out_dim = parameters["HiddenSize"] if fc_layer < M-1 else num_classes
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.Dropout(parameters["Dropout"]))
        in_dim = out_dim

    return nn.Sequential(*layers)
