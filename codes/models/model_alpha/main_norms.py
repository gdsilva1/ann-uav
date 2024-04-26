import torch
from torch import nn
from torch.utils.data import random_split, DataLoader

from scipy.io import loadmat

from ghelpers.misc import preprocessing_from_matlab, plot_loss_function
from gdataset.states_forces import StatesAndForces
from gmodels.models import gNN
from ghelpers.training import train, test

import os
from alive_progress import alive_bar


# Setting GPU if avaiable
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
print("=" * 40)
print("|" + f"DEVICE: {device}".center(38) + "|")
print(40 * "=", end="\n\n")

# Setting sedd to reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Path to the MATLAB files
path_to_states = "/home/gabriel/Documents/matlab_files/xs_all.mat"
path_to_forces = "/home/gabriel/Documents/matlab_files/tau_all.mat"

# Raw files from MATLAB
states_raw = loadmat(path_to_states)
forces_raw = loadmat(path_to_forces)

# List with 1000 matrices for states and forces
_, states_norms = preprocessing_from_matlab(states_raw, n=100)
_, forces_norms = preprocessing_from_matlab(forces_raw, n=100)

# Standard format for Torch dataset
dataset = StatesAndForces(states_norms, forces_norms)

# Hyper-parameters
TRAIN_PROP = 0.8
BATCH_SIZE = 1
INPUT_LAYER = 12
HIDDEN_LAYERS = 128
OUTPUT_LAYER = 4
ACTIVATION_FUNCTION = nn.ReLU()
LOSS_FUNCTION = nn.MSELoss()
EPOCHS = 500

# Splititng data into train and test
train_data, test_data = random_split(dataset, [TRAIN_PROP, 1 - TRAIN_PROP])

# Standard format for Torch dataloader (iterable)
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

print(40 * "=")
print("|" + f"Lenght of train data: {len(train_dataloader)}".center(38) + "|")
print("|" + f"Lenght of test data: {len(test_dataloader)}".center(38) + "|")
print(40 * "=", end="\n\n")

# Creating the model
model = gNN(
    input_layer=INPUT_LAYER,
    hidden_layers=HIDDEN_LAYERS,
    output_layer=OUTPUT_LAYER,
    activation_function=ACTIVATION_FUNCTION,
).to(device)
loss_fn = LOSS_FUNCTION
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training the model
epochs = EPOCHS
loss_overall = []
loss_overall_test = []

with alive_bar(
    EPOCHS,
    title="TRAINING",
    stats=False,
    length=10,
    spinner="classic",
    bar="brackets",
    elapsed=False,
) as bar:
    for epoch in range(epochs):
        loss = train(train_dataloader, model, loss_fn, optimizer, device)
        test_loss = test(test_dataloader, model, loss_fn, device)
        bar.text(f"| Train loss: {loss:.6f} | Test loss: {test_loss:.6f}")
        loss_overall.append(loss.item())
        loss_overall_test.append(test_loss.item())
        bar()

plot_loss_function(epochs=EPOCHS, loss=loss_overall, loss_test=loss_overall_test, n=2)

print("\nDone!")

# Saving the model
path_to_model = "./models/"
if not os.path.exists(path_to_model):
    os.mkdir(path_to_model)

path_saved = os.path.join(path_to_model, "model_norms.pt")
torch.save(model.state_dict(), path_saved)
