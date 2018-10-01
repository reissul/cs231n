import matplotlib.pyplot as plt
import numpy as np
import os
import simplejson
from builtins import range
from math import sqrt, ceil
from os.path import exists, join
from past.builtins import xrange

from visdom import Visdom

from cs231n.train_utils import *

class Visualizer(object):
    def __init__(self, logdir, port=6006):
        self.vis = Visdom(port=port)
        assert self.vis.check_connection(), 'No connection could be formed quickly'
        self.vis.close()
        plt.close('all')
        self.matplot_win = None
        self.logdir = logdir
        self.data = {}
        self.windows = {}
        self.table_cols = []
        self.table_data = {}
    def __del__(self):
        plt.close('all')
        del self.vis
    def update(self, table=False):
        for model in os.listdir(self.logdir):
            if "model" not in model: continue
            # Load *new* data from the model directory.
            if model not in self.data:
                self.data[model] = {LOSSES: [], TRAIN_ACCS: [], VAL_ACCS: []}
                self.windows[model] = dict(loss=None, accs=None)
            new_x, new_y = {}, {}
            for k in [LOSSES, TRAIN_ACCS, VAL_ACCS]:
                fname = join(self.logdir, model, k)
                if not exists(fname): continue
                d = list(map(float, open(fname).readlines()))
                new_x[k], new_y[k] = [], []
                N = len(self.data[model][k])
                for i in range(N, len(d)):
                    new_x[k].append(i)
                    new_y[k].append(d[i])
                    self.data[model][k].append(d[i])
            # Load table data for this model.
            if table and self.data[model][TRAIN_ACCS]:
                fname = join(self.logdir, model, "arch.json")
                parameters = simplejson.load(open(fname))
                id = int(model.replace("model", ""))
                p = sorted(parameters.items())
                self.table_cols = ["id"] + sorted(parameters.keys()) + \
                    ["loss", "train acc", "val acc"]
                row = [id]
                row += [parameters[col] for col in self.table_cols if col in parameters]
                row += [self.data[model][LOSSES][-1]]
                row += [self.data[model][TRAIN_ACCS][-1]]
                row += [self.data[model][VAL_ACCS][-1]]
                self.table_data[model] = row
            # Continue if no new data or not working.
            if not new_y[LOSSES] or not is_working(self.data[model][LOSSES]):
                continue
            # Visualize plots for this model.
            if LOSSES in new_x and new_x[LOSSES]:
                opts = {
                    "title": model,
                    #"xlabel": "iteration",
                    "ylabel": "loss",
                    "xtickmin": 0,
                    "xtickmax": (1 + len(self.data[model][LOSSES]) // 100) * 100,
                    "ytickmin": 0,
                    "ytickmax": 5
                }
                w = self.windows[model]["loss"]
                if not w:
                    y = self.data[model][LOSSES]
                    x = list(range(len(y)))
                    self.windows[model]["loss"] = self.vis.line(X=x, Y=y, opts=opts)
                else:
                    self.vis.line(X=new_x[LOSSES], Y=new_y[LOSSES], opts=opts,
                                  win=w, update="append")
            if TRAIN_ACCS in new_x and new_x[TRAIN_ACCS]:
                opts = {
                    "title": model,
                    #"xlabel": "iteration",
                    "ylabel": "accuracy",
                    "xtickmin": 0,
                    "xtickmax": (1 + len(self.data[model][TRAIN_ACCS]) // 10) * 10,
                    "ytickmin": 0,
                    "ytickmax": 100,
                    "legend": ["Train", "Val"]
                }
                update = "append" if self.windows[model]["accs"] else None
                w = self.windows[model]["accs"]
                if not w:
                    y = list(zip(self.data[model][TRAIN_ACCS],
                                 self.data[model][VAL_ACCS]))
                    x = list(range(len(y)))
                    self.windows[model]["accs"] = self.vis.line(X=x, Y=y, opts=opts)
                else:
                    self.vis.line(X=list(zip(new_x[TRAIN_ACCS], new_x[VAL_ACCS])),
                                  Y=list(zip(new_y[TRAIN_ACCS], new_y[VAL_ACCS])),
                                  opts=opts, win=w, update="append")
        # Visualize table data for all models.
        if table and self.table_data.values():
            # Sort by val acc.
            data = sorted(self.table_data.values(), key=lambda x: -x[-1])
            # Update table in visom.
            N, M = np.array(data).shape
            fig, ax = plt.subplots()
            fig.set_figheight(max(5, N/2)) # ewww
            fig.set_figwidth(20)
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            ax.table(cellText=data, colLabels=self.table_cols, loc='center')
            self.matplot_win = self.vis.matplot(plt, win=self.matplot_win)

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G

def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H,W,C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G
