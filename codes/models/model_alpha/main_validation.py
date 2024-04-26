from scipy.io import loadmat
from matplotlib import pyplot as plt
from gmodels.models import gNN
from ghelpers.misc import preprocessing_from_matlab, denormalize, plot_forces
import torch

# Path to the MATLAB files
path_to_states = "/home/gabriel/Documents/matlab_files/xs_all.mat"
path_to_forces = "/home/gabriel/Documents/matlab_files/tau_all.mat"

# Raw files from MATLAB
states_raw = loadmat(path_to_states)
forces_raw = loadmat(path_to_forces)

# Original force and state from matlab
random_force_denorm = forces_raw["tau_all"][950][0].T
random_state_denorm = states_raw["xs_all"][950][0].T

# Preprocessing from MATLAB
states, states_norms = preprocessing_from_matlab(states_raw)
forces, forces_norms = preprocessing_from_matlab(forces_raw)

# Random state and force
random_states, random_states_norm = states[950], states_norms[950]
random_forces, random_forces_norm = forces[950], forces_norms[950]

# Loading models
model = gNN(input_layer=12, hidden_layers=128, output_layer=4)
model.load_state_dict(torch.load("./models/model.pt"))
model_norms = gNN(input_layer=12, hidden_layers=128, output_layer=4)
model_norms.load_state_dict(torch.load("./models/model_norms.pt"))

# Getting values from the model
model.eval()
with torch.inference_mode():
    normalized_forces = model(torch.tensor(random_states, dtype=torch.float32))

model_norms.eval()
with torch.inference_mode():
    forces_norms = model_norms(torch.tensor(random_states_norm, dtype=torch.float32))

# Denormalizing forces
forces = denormalize(normalized_forces, forces_norms)

# Plotting comparison
plt.style.use("duarte.mplstyle")
plot_forces(random_force_denorm, forces=forces, n=100)
plot_forces(normalized_forces, n=100)