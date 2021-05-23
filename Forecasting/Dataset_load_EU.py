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

path = "F:\GitHub\COVID-19-EU\data-by-region"
os.environ['PATH'] += ':' + path

EU_df = pd.read_csv(path + '\jrc-covid-19-all-days-by-regions.csv')

country = 'Germany'

index = EU_df.index
region_idx = index[EU_df['CountryName'].str.contains(country)].tolist()


EU_df_new = EU_df.iloc[region_idx, :][['Date','CountryName','Region','CumulativePositive']]


Region_list = EU_df_new.Region.unique().tolist()
Dates = EU_df_new.Date.unique().tolist()


A = pd.DataFrame(index=Region_list, columns= Dates)

for _date in Dates:
    B = EU_df_new.loc[EU_df_new['Date'] == _date][['Region', 'CumulativePositive']]
    B = B.set_index('Region')
    A.loc[B.index, _date] = B.values.reshape(-1)

A


