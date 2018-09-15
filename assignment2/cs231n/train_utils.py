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
    @property
    def working(self):
        return (self.val_accs[-1] - self.val_acs[0]) > 5.

class ChoiceSampler(object):
    def __init__self(self, choices):
        self.bad = np.zeros(len(choices))
        self.tot = np.zeros(len(choices))
        self.choices = choices
    def sample(self):
        good = (self.bad / (self.tot+1)) < -1#0.75
        assert(np.sum(good))
        ind = np.choice(np.nonzero(good)[0], 1)
        return self.choices[ind]
    def update(self, bad, val):
        ind = [i for (i,v) in enumerate(self.choices) if abs(v-val) < 1e-9][0]
        self.bad[ind] += bad
        self.tot[ind] += 1

class HyperOpt(object):
    """
    A hyperparameter optimizer. Includes individual parameter samplers,
    training routines, and coarse/fine strategies.
    
    Example usage:

    samplers = {
      "FilterSize": ChoiceSampler([3, 5]),
      "FilterCount": ChoiceSampler([8, 32, 128]),
    }
    ho = HyperOpt(samplers)
    while True:
      ho.step()
      print("Best parameters:", ho.best_parameters)
    """
    def __init__(self, samplers, constructor, trainer, loader_train, loader_val,
                 max_active=None, coarse_its=None, fine_epochs=10,
                 coarse_fine="contract"):
        """
        Construct a HyperOpt object.
        
        Required arguments:
        - samplers: Dictionary from parameter name to parameter Sampler
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
        self.samplers = samplers
        self.constructor = constructor
        self.trainer = trainer
        self.max_active = max_active
        self.coarse_its = coarse_its
        self.fine_epochs = fine_epochs
        self.coarse_fine = coarse_finne
        self.fine = False # begin coarse
        self.active = queue.PriorityQueue() # begin with no active
        self.results = [] # begin with no results
    def step(self):
        parameters = self.sample()
        model = self.constructor(parameters, 32) # TODO: don't hard code?
        if coarse_fine == "contract":
            # See if we need to reset fine.
            if not self.fine and len(self.active) == self.max_active:
                self.fine = True
            elif self.fine and not self.active:
                self.fine = False
            # If we are doing coarse search.
            if not self.fine:
                # Find the right learning rate (if there is one).
                lrs = []
                while len(lrs) < 3:
                    lr = 10 ** np.uniform(-6, 1)
                    if lr in lrs: continue
                    optimizer = optim.SGD(model.parameters(), lr=lr,
                                          momentum=0.9, nesterov=True)
                    history = self.trainer(model, optimizer, self.loader_train,
                                           self.loader_val, epochs=1,
                                           its=self.coarse_its, eval_its=16)
                    # If it's working, add it to our active set and break.
                    if history.working:
                        self.active.put((-history.val_acc, model, optimizer, history))
                        break
            # If we are doing fine search.
            else:
                _, model, optimizer, history = self.active.get()
                history = self.trainer(model, optimizer, self.loader_train,
                                       self.loader_val, epochs=self.fine_epochs,
                                       eval_its=16)
                self.trained.put((-history.val_acc, model, history))
        else:
            pass # TODO: other coarse/fine strategies.

    def sample(self):
        return dict([(n, p.sample()) for (n, s) in self.samplers.items()])

    def update(parameters, hist):
        [self.samplers[n].update(hist.bad, v) for (n, v) in parameters.items()]
