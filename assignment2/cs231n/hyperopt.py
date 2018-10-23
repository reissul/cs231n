import inspect
import logging
import os
import simplejson
from numpy.random import randint
from os.path import join, exists
from random import choice

import torch.optim as optim
from visdom import Visdom

from cs231n.train_utils import *
from cs231n.vis_utils import *
from cs231n.parameter_server import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class HyperOpt(object):
    
    def __init__(self, choices, loader_train, loader_val, logdir,
                 max_coarse=3, vis=True, verbose=True, coarse_its=None,
                 fine_epochs=10, device=torch.device('cpu')):
        self.parameter_server = ParameterServer(choices, logdir)
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.logdir = logdir
        if not exists(logdir):
            os.makedirs(logdir)
        self.max_coarse = max_coarse
        self.vis = Visualizer(logdir) if vis else None
        self.verbose = verbose
        self.coarse_its = coarse_its
        self.fine_epochs = fine_epochs
        self.device = device
        
        self.coarse_trained = []
        self.fine_trained = []
        
        self.load()

    def load(self):
        for model_fname in os.listdir(self.logdir):
            if "model" not in model_fname: continue
            model_dir = join(self.logdir, model_fname)
            # Load the losses and continue if not working.
            fname = join(model_dir, LOSSES)
            if exists(fname):
                losses = list(map(float, open(fname).readlines()))
                if not is_working(losses):
                    continue
            # Load last checkpoint
            check_fnames = [v for v in os.listdir(model_dir) if "check" in v]
            if not check_fnames: continue
            check_fname = join(model_dir, sorted(check_fnames)[-1])
            check = Checkpoint()
            check.load(check_fname)
            check.logdir = model_dir
            if check.epoch < self.fine_epochs:
                self.coarse_trained.append(check)
            else:
                self.fine_trained.append(check)
        
    def optimize(self):
        fname = inspect.stack()[0][3]
        logger.debug('%s: entering' % (fname))
        fine = len(self.coarse_trained) >= self.max_coarse
        while self.parameter_server.serving or self.coarse_trained:
            # If the coarse search is done, move to fine.
            if not fine and (len(self.coarse_trained) == self.max_coarse \
                                 or not self.parameter_server.serving):
                fine = True
            # If the fine search is done, move to coarse.
            elif fine and not self.coarse_trained:
                fine = False
            # Step
            if not fine:
                self.coarse_step()
            else:
                self.fine_step()

    def coarse_step(self):
        fname = inspect.stack()[0][3]
        logger.debug('%s: starting' % (fname))
        
        # Sample parameters.
        parameters = self.parameter_server.serve()
                
        # Find the most aggressive learning rate that works.
        for lr in [10**v for v in range(-3, -5, -1)]:
            model = construct_model(parameters, (3,32,32), 10) # TODO: don't hard code?
            model_name = "model%03d" % (len(os.listdir(self.logdir)) + 1)
            model_logdir = join(self.logdir, model_name)
            logger.debug('%s: %s with lr=%.3f' % (fname, model_name, lr))
            logger.debug('%s: %s' % (fname, str(model)))
            if not exists(model_logdir):
                os.makedirs(model_logdir)
            parameters["Rate"] = lr
            simplejson.dump(parameters, open(join(model_logdir, "arch.json"), "w"))
            #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
            #                      nesterov=True)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-9)
            losses, checkpoint = train(self.loader_train, self.loader_val, model, optimizer, 
                                       model_logdir, verbose=self.verbose, vis=self.vis, epochs=1,
                                       iterations=self.coarse_its, eval_iterations=16,
                                       device=self.device)
            checkpoint.logdir = model_logdir
            if is_working(losses):
                self.coarse_trained.append(checkpoint)
                break
    
    def fine_step(self):
        fname = inspect.stack()[0][3]
        self.coarse_trained = sorted(self.coarse_trained, key=lambda x: -x.acc)
        check = self.coarse_trained.pop(0)
        logger.debug('%s: %s' % (fname, check.logdir.split("/")[-1]))
        check = train(self.loader_train, self.loader_val, check.model, check.optimizer,
                      check.logdir, verbose=self.verbose, vis=self.vis,
                      epochs=self.fine_epochs, eval_iterations=16, device=self.device)
        self.fine_trained.append(check)
