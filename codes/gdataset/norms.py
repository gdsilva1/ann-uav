import torch
from torch.utils.data import Dataset


class Norms(Dataset):

    def __init__(self, states_norms, forces_norms):
        super().__init__()
        self.states_norms = states_norms
        self.forces_norms = forces_norms
        if len(states_norms) == len(forces_norms):
            self.length = len(states_norms)
        else:
            raise ValueError("States and forces must be the same length.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        state_norms = torch.tensor(self.states_norms[index], dtype=torch.float32)
        force_norms = torch.tensor(self.forces_norms[index], dtype=torch.float32)

        return state_norms, force_norms
