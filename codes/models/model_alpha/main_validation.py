from scipy.io import loadmat
from matplotlib import pyplot as plt
from numpy import linspace, array
from gmodels.models import gNN
from ghelpers.misc import preprocessing_from_matlab
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
random_state, random_state_norm = states[950], states_norms[950]
random_force, random_force_norm = forces[950], forces_norms[950]

# Loading models
model = gNN(input_layer=12, hidden_layers=128, output_layer=4)
model.load_state_dict(torch.load("./models/model.pt"))
model_norms = gNN(input_layer=12, hidden_layers=128, output_layer=4)
model_norms.load_state_dict(torch.load("./models/model_norms.pt"))

# Getting values from the model
model.eval()
with torch.inference_mode():
    normalized_forces = model(torch.tensor(random_state, dtype=torch.float32))

model_norms.eval()
with torch.inference_mode():
    forces_norms = model_norms(torch.tensor(random_state_norm, dtype=torch.float32))


forces = []
for i in range(normalized_forces.shape[-1]):
    force_denorm = normalized_forces[:,i]*forces_norms[i]
    force_denorm = force_denorm.numpy()
    forces.append(force_denorm)
forces = array(forces).T

def plot_forces():
    t = linspace(0,30,len(random_force))
    fig, axs = plt.subplots(4,1)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(t, random_force[:,i], label="Real")
        ax.plot(t, forces[:,i], label="Pred")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"$U_{i+1}$")
        fig.tight_layout()
    plt.show()
        
plot_forces()