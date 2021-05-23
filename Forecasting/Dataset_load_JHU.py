import os
from datetime import datetime

# # machine learning
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import TensorBoard
#
# data manipulation and signal processing
import math
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import scipy.stats as ss

#
# # plots
# import seaborn as sns
# import matplotlib
# import matplotlib.pyplot as plt
# import folium
# import pydot

path = "F:\GitHub\COVID-19-JHU\csse_covid_19_data\csse_covid_19_daily_reports"
os.environ['PATH'] += ':' + path

filenames = os.listdir(path)
_names = ['.gitignore', 'README.md']

for k in range(len(_names)):
    if _names[k] in filenames:
        filenames.remove(_names[k])

names_new = []

for word in filenames:
    _date = datetime.strptime(word, '%m-%d-%Y.csv')
    names_new.append(_date.strftime("%Y-%m-%d"))

names_sorted = sorted(names_new)
files_sorted = []
for word in names_sorted:
    _date = datetime.strptime(word, '%Y-%m-%d')
    files_sorted.append(_date.strftime("%m-%d-%Y.csv"))

_df0 = pd.read_csv(path + '\\' + files_sorted[100])

country = "Australia"

if 'Province_State' in _df0:                                        #if provincial data exists
    _df1 = _df0[['Province_State', 'Country_Region', 'Confirmed']]
    if _df0['Country_Region'].str.contains(country).any():          #if country exists
        index = _df0.index
        region_idx = index[_df0['Country_Region'].str.contains(country)].tolist()
        if len(region_idx) > 1:
            print(region_idx)

