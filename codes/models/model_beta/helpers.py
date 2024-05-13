import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize as n

def preprocessing_from_matlab_new(matlab_file_path: str, flat: bool = False) -> list[np.ndarray]:
    data: dict = loadmat(matlab_file_path)
    dict_keys = list(data.keys())
    last_key: str = dict_keys[-1]
    data_all_messed = data[last_key]

    organized_data: list = []
    for row in data_all_messed:
        organized_data.append(row[0].T)
    
    if flat:
        flattened_data = []
        for matrix in organized_data:
            flattened_data.append(matrix.flatten())
        return flattened_data
    else:
        return organized_data

def normalize_new(data: list[np.ndarray]) -> list[np.ndarray]:
    data_normalized: list = []
    data_norms: list = []
    for row in data:
        var_normalized, var_norm = n(X=row, norm="l2", return_norm=True, axis=0)
        data_normalized.append(var_normalized)
        data_norms.append(var_norm)
    return data_normalized, data_norms
