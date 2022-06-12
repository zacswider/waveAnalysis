import os                                       
import sys 
import timeit
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from waveanalysismods.customgui import BaseGUI, RollingGUI
from waveanalysismods.processor import SignalProcessor, SignalProcessor_new

impath = '/Users/bementmbp/Desktop/Scripts/waveAnalysis/test_data/1_Group1.tif'

new = SignalProcessor_new(impath, kern = 20, step = 20, roll=True, roll_size=25, roll_by=5)

print(new.means.shape)

new.calc_ACF()

print(new.acfs.shape)
print(new.periods.shape)


#plt.plot(new.acfs[0,0,0,:])
#print(new.periods[0,0,0])

cf_plots = new.plot_mean_CF()
ch1 = cf_plots['Ch1 Mean ACF']

measurements, summary = new.summarize_results()

print(measurements)

for key, value in summary.items():
    print(key, value)



#plt.show()