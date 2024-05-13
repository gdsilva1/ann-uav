from sklearn.preprocessing import normalize
from scipy.io import loadmat
from matplotlib import pyplot as plt
from numpy import array, linspace, ndarray
from typing import Any


def preprocessing_from_matlab(data: dict[Any], n: int = 0) -> list[list]:
    """Return the matlab trajectories normalized and ready to use.

    Parameters
    ----------
    data : dict[Any]
        The MATLAB loaded file (with scipy.io.loadmat)
    n : int, optional
        Number of samples that are going to be used., by default 0

    Returns
    -------
    list[list]
        The data preprocessed.
    """
    key = list(data.keys())[-1]
    all_data = data[key].squeeze()
    all_data_normalized: list = []
    all_data_norms: list = []

    for data in all_data:
        data_normalized, data_norms = normalize(X=data.T[n:], axis=0, return_norm=True)
        all_data_normalized.append(data_normalized)
        all_data_norms.append(data_norms)

    return all_data_normalized, all_data_norms


def denormalize(normalized_forces: list | ndarray, forces_norms: list | ndarray) -> ndarray:
    """Denormalize the forces.

    Parameters
    ----------
    normalized_forces : list | ndarray
        The normalized forces.
    forces_norms : list | ndarray
        The norms related to the normalized forces.

    Returns
    -------
    ndarray
        The denormalized forces.
    """
    forces: list = []
    for i in range(normalized_forces.shape[-1]):
        force_denorm: list | ndarray = normalized_forces[:, i] * forces_norms[i]
        force_denorm = force_denorm.numpy()
        forces.append(force_denorm)
    forces: ndarray = array(forces).T
    return forces


def plot_loss_function(
    epochs: int, loss: list | ndarray, loss_test: list | ndarray, n: int
) -> None:
    """Return the plot of the loss function vs epochs. 

    Parameters
    ----------
    epochs : int
        Number of epochs.
    loss : list | ndarray
        Loss number per epoch.
    loss_test : list | ndarray
        Test loss number per epoch.
    n : int
        Which neural network is being used.
    """
    CM = 1 / 2.54
    plt.style.use("duarte.mplstyle")
    fig, ax = plt.subplots(figsize=(7 * CM, 7 * CM))

    ax.plot(range(epochs), loss, label="Train loss")
    ax.plot(range(epochs), loss_test, label="Test loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, 60)
    ax.legend()
    ax.set_title(f"ANN {n} Loss Function", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"./figures/ann{n}_loss_function.pdf")


def plot_forces(random_forces, n, forces=None):
    t = linspace(0, 30, len(random_forces))
    CM = 1 / 2.54
    fig, axs = plt.subplots(4, 1, figsize=(16 * CM, 20 * CM))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(t[n:], random_forces[:, i][n:], label="Real")
        if forces is not None:
            ax.plot(t[n:], forces[:, i][n:], label="Pred")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"$U_{i+1}$")
        if forces is not None:
            ax.legend()
    fig.tight_layout()
    if forces is not None:
        fig.savefig("./figures/comparison.pdf")
    else:
        fig.savefig("./figures/normalized_forces.pdf")


from scipy.io import loadmat
from sklearn.preprocessing import normalize as n
import numpy as np

def preprocessing_from_matlab_new(matlab_file_path: str) -> list[np.ndarray]:
    data: dict = loadmat(matlab_file_path)
    dict_keys = list(data.keys())
    last_key: str = dict_keys[-1]
    data_all_messed = data[last_key]

    data_organized: list = []
    for row in data_all_messed:
        data_organized.append(row[0].T)
    return data_organized

def normalize_new(data: list[np.ndarray]) -> list[np.ndarray]:
    data_normalized: list = []
    data_norms: list = []
    for row in data:
        var_normalized, var_norm = n(X=row, norm="l2", return_norm=True, axis=0)
        data_normalized.append(var_normalized)
        data_norms.append(var_norm)
    return data_normalized, data_norms
