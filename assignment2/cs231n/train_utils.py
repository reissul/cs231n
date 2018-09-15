def main():
    
    im_shape = (3, 32, 32)
    num_classes = 10
    
    samplers = {
        "Architecture": ChoiceSampler(["conv-relu-pool",
                                       "batchnorm-relu-conv"])
        "FilterSize": ChoiceSampler([3, 5, 7]),
        "FilterCount": ChoiceSampler([4, 32, 127]),
        "Stride": ChoiceSampler([1, 2]),
        "N": ChoiceSampler([3, 5, 10]),
        "M": ChoiceSampler([0, 2]),
        "Dropout": ChoiceSampler([0.25, 0.5, 0.95])
        }
    
    master = Master(samplers)
    
    while True:
        parameters = master.sample_parameters()
        model = construct_model(parameters, im_shape)
        history = train(model)
        master.update(parameters, history, model)
        master.save()
        best = master.get_best()
        print("best model acc = %.2f" % best.acc)

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

class Master(object):
    def __init__(self, samplers):
        self.samplers = samplers
        self.results = []
    def sample_parameters(self):
        return dict([(n, p.sample()) for (n, s) in self.samplers.items()])
    def update(parameters, history, model):
        # Update results with history and model info.
        self.results.append((history.val_acc, history, model))
        # Update samplers.
        for (name, val) in parameters.items():
            self.samplers[name].update(history.bad, val)


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
