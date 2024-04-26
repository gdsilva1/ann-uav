from sklearn.preprocessing import normalize
from scipy.io import loadmat
from matplotlib import pyplot as plt
from numpy import array, linspace, ndarray


def preprocessing_from_matlab(data: dict, n: int = 0) -> list[list]:
    key = list(data.keys())[-1]
    all_data = data[key].squeeze()
    all_data_normalized: list = []
    all_data_norms: list = []

    for data in all_data:
        data_normalized, data_norms = normalize(X=data.T[n:], axis=0, return_norm=True)
        all_data_normalized.append(data_normalized)
        all_data_norms.append(data_norms)

    return all_data_normalized, all_data_norms


def denormalize(normalized_forces, forces_norms):
    forces = []
    for i in range(normalized_forces.shape[-1]):
        force_denorm = normalized_forces[:, i] * forces_norms[i]
        force_denorm = force_denorm.numpy()
        forces.append(force_denorm)
    forces = array(forces).T
    return forces


def plot_loss_function(
    epochs: int, loss: list | ndarray, loss_test: list | ndarray, n: int
) -> None:
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


def preprocessing_from_matlab_new(matlab_file_path):
    data_file_raw = loadmat(matlab_file_path)
    data_name = list(data_file_raw.keys())[-1]
    data = data_file_raw[data_name].squeeze()
    for variable in data:
        yield variable.T


def normalize_data_new(data):
    for item in data:
        yield normalize(item, return_norm=True, axis=0)
