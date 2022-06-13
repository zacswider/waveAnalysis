from importlib.metadata import files
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
from waveanalysismods.processor import TotalSignalProcessor
from waveanalysismods.rollingprocessor import RollingSignalProcessor

def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


impath = '/Users/bementmbp/Desktop/Scripts/waveAnalysis/test_data/1_Group1.tif'
#impath = '/Users/bementmbp/Desktop/BementLab/2_Projects/29_tripleWavePaper/Figures/Figure01/Fig01A-C/201015_Live_SFC_Aegg_GFP-pGBD_mCh_Utr647_E04-T01_Max.tif'

new = RollingSignalProcessor(impath, kern = 20, step = 20, roll_size = 25, roll_by = 5)
new.calc_ACF()
new.calc_CCF()
new.calc_peak_props()
df = new.summarize_file()
print(df)
