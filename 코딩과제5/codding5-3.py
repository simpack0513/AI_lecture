import csv
import torch
import numpy as np
wine_path = "winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

wineq = torch.from_numpy(wineq_numpy)
print(wineq.shape, wineq.dtype)
