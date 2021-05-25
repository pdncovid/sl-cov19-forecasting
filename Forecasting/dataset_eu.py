import os
import sys
import numpy as np

import matplotlib.pyplot as plt

from Forecasting.utils.data_loader import load_data_eu
from Forecasting.utils.smoothing_functions import O_LPF, NO_LPF

country = 'Italy'
d = load_data_eu(country=country, path=sys.path[-1] + '\Datasets', provinces=True)
region_names = d["region_names"]
confirmed_cases = d["confirmed_cases"]

daily_cases = d["daily_cases"]
START_DATE = d["START_DATE"]
n_regions = d["n_regions"]
daily_cases[daily_cases < 0] = 0

# %% OVER FILTERING AND UNDER FILTERING PROBLEM
# daily_filtered_na = NO_LPF(daily_cases, datatype='daily', order=10, cutoff=0.088, region_names=region_names)
# daily_filtered_na2 = NO_LPF(daily_cases, datatype='daily', order=10, cutoff=0.015, region_names=region_names)

# %% OPTIMAL FILTERING
# daily_filtered = O_LPF(daily_cases, datatype='daily', order=3, R_weight=1.0, EIG_weight=1.1, corr=False,
#                        region_names=region_names, view=True)

