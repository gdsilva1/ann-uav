# Setup
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import POO as f
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from tqdm.auto import trange, tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
# device = "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
plt.style.use("duarte")


# Loading the MATLAB files.
forces = loadmat("../../../../tau_all_noise.mat")
states = loadmat("../../../../xs_all_noise.mat")

forces_preprocessed = f.preprocessing_from_matlab(forces, "tau_all")
states_preprocessed = f.preprocessing_from_matlab(states, "xs_all")



cm = 1/2.54
rs = states_preprocessed[random.randrange(10,900)]
fig_trajectory, ax_trajectory = plt.subplots(subplot_kw={'projection': '3d'},
                                             constrained_layout=True,
                                             figsize=(16*cm, 12*cm))

ax_trajectory.plot(rs[:,0][500:], rs[:,1][500:], rs[:,2][500:])
ax_trajectory.set_xlabel("$x(t)$")
ax_trajectory.set_ylabel("$y(t)$")
ax_trajectory.set_zlabel("$z(t)$")
ax_trajectory.invert_zaxis()

fig_trajectory.savefig("/home/gabriel/Documentos/deep-learning/report/figures/4results/uav/trajectory.pdf", backend="pgf")

forces_normalized = 
dataset = StatesAndForces(all_states, all_forces)
