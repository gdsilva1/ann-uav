# %% [markdown]
# # Model 5

# %% [markdown]
# ## Data preprocessing

# %%
# Setup
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")

# %%
# Loading the MATLAB files.
xs = loadmat('xs_all.mat')
tau = loadmat('tau_all.mat')

# %%
class LoadXsTau(Dataset):

    def __init__(self, xs: dict, tau: dict, normalize: bool = True) -> None:
        super().__init__()
        self.tau_all = tau['tau_all'].squeeze() 
        self.xs_all = xs['xs_all'].squeeze()
        self.length = len(self.xs_all)
        self.normalize = normalize

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        xs = self.xs_all[index]
        tau = self.tau_all[index]

        if self.normalize:
            xs = normalize(xs, axis=0)
            tau = normalize(tau, axis=0)

        xs = torch.tensor(xs.T, dtype=torch.float32)
        tau = torch.tensor(tau.T, dtype=torch.float32)

        return xs, tau

# %%
dataset = LoadXsTau(xs, tau, normalize=True)
train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_dataloader= DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader= DataLoader(test_data, batch_size=32, shuffle=True)

print(f"Lenght of train data: {len(train_dataloader)}")
print(f"Lenght of test data: {len(test_dataloader)}")

# %% [markdown]
# ## Creating the Model

# %%
class ModelA(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.layers(x)

# %%
model = ModelA().to(device)
model.state_dict()

# %%
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 400 == 0:
        #     print(f"Looked at {batch * len(X)}/{len(dataloader.dataset)} samples")

        return loss

def test(dataloader, model, loss_fn):
    model.eval()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

    with torch.inference_mode():
        test_pred = model(X)
        test_loss = loss_fn(test_pred, y)

    return test_loss


# %%
epochs = 100
loss_overall = []
loss_overall_test = []

for epoch in tqdm(range(epochs), leave=False):
    loss = train(train_dataloader, model, loss_fn, optimizer)
    test_loss = test(test_dataloader, model, loss_fn)
    if epoch % 20 == 0:
        print(f"EPOCH {epoch+1}")
        print(f"Epoch: {epoch+1} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")
        print("-"*50)
    loss_overall.append(loss.item())
    loss_overall_test.append(test_loss.item())
    
print("Done!")

# %%
# %matplotlib ipympl
t = np.linspace(1, epochs, epochs)

plt.plot(t, loss_overall, label='Train loss')
plt.plot(t, loss_overall_test, label='Test loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%



