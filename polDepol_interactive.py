import numpy as np
import pandas as pd  
import seaborn as sns
import skimage.io as skio  
import skimage.transform as skit
import scipy.signal as sig                        
import matplotlib.pyplot as plt    
from matplotlib.widgets import Slider
from tkinter.filedialog import askdirectory     
from matplotlib.animation import FuncAnimation   

rawFilePath = "/Volumes/FlashSSD/210930_Live_Flvw_Ocyte_mVenus-ssra_caax_mScl-Chm1-GAP_mVenus_SSrA-caax/210930_Live_Flvw_Ocyte_mVenus-ssra_caax_mScl-Chm1-GAP_mVenus_SSrA-caax_E03_T02_MaxSmooth.tif"
image = skio.imread(rawFilePath).astype('float64') #array of shape (frames, y, x)
diffNumber = 5
windowSize = 5
scaleFactor = 10

print(image.shape)
raw = skit.downscale_local_mean(image, (1,scaleFactor,scaleFactor))
print(raw.shape)

def calcLines(rawData, diffNum, window):
    diff = np.subtract(rawData[diffNum:], rawData[:-diffNum])
    polZAxis =  np.nanmean(np.where(diff>0, diff, np.nan), axis=(1,2))
    depolZAxis = np.abs(np.nanmean(np.where(diff<0, diff, np.nan), axis=(1,2)))
    numPoints = int(rawData.shape[0]-diffNum)
    diffXAxis = np.linspace(1, numPoints, numPoints)
    rollPol = np.convolve(polZAxis, np.ones(window), 'valid') / window      #returns rolling average array
    rollDepol = np.convolve(depolZAxis, np.ones(window), 'valid') / window      #returns rolling average array
    numPointsRoll = int(rawData.shape[0]-diffNum-window//2)
    rollXAxis = np.linspace((1+window//2), (numPointsRoll-window//2), (numPointsRoll-window//2))
    rawMean = np.nanmean(rawData, axis=(1,2))
    return(diffXAxis, polZAxis, depolZAxis, rollXAxis, rollPol, rollDepol, rawMean)

fig = plt.figure(figsize=(7, 5))        #figure object
ax = fig.add_subplot(111)               #Create main axis; 111=row,column,position. Not strictly necessary with only one subplot
fig.subplots_adjust(bottom=0.2, top=0.75)   #position as a fraction of the figure width
diffAx = fig.add_axes([0.3, 0.85, 0.4, 0.05])    #rectangle of size [x0, y0, width, height]
rollAx = fig.add_axes([0.3, 0.92, 0.4, 0.05])
rollValues = np.array([1,3,5,7,9,11,13,15])
diffValues = np.linspace(1,15,15)
try:
    diffSlider = Slider(ax=diffAx, label='Frames to difference ', valmin=1, valmax=15, valinit=5, valfmt=' %1.1f Frames', valstep=diffValues, facecolor='#cc7000')
except ValueError:
    diffSlider = Slider(ax=diffAx, label='Frames to difference ', valmin=1, valmax=15, valinit=5, valfmt=' %1.1f Frames', valstep=diffValues.all(), facecolor='#cc7000')
try:
    rollSlider = Slider(ax=rollAx, label='Frames to average ', valmin=1, valmax=15, valinit=5, valfmt='%i Frames', valstep=rollValues, facecolor='#cc7000')
except ValueError:
    rollSlider = Slider(ax=rollAx, label='Frames to average ', valmin=1, valmax=15, valinit=5, valfmt='%i Frames', valstep=rollValues.all(), facecolor='#cc7000')
ax.set_ylabel('relative assembly and disassembly')
ax.set_xlabel('time (frames)')

xAxisDots, polDotVals, depolDotVals, xAxisRoll, polRoll, depolRoll, rawMean = calcLines(raw, diffNumber, windowSize)
polDots, = ax.plot(xAxisDots, polDotVals, color='deepskyblue', marker='o', linestyle="", markersize=2, alpha=0.25)
depolDots, = ax.plot(xAxisDots, depolDotVals, color='darkorange', marker='o', linestyle="", markersize=2, alpha=0.25)
polLine, = ax.plot(xAxisRoll, polRoll, color='deepskyblue', label='recent F-actin assembly')
depolLine, = ax.plot(xAxisRoll, depolRoll, color='darkorange', label='recent F-actin disassembly')
#ax.plot(rawMean)
ax.legend(loc='upper right', fontsize='small', frameon=False, ncol=1)

def update(val):
    d = int(diffSlider.val)
    r = int(rollSlider.val)
    xAxisDots, polDotVals, depolDotVals, xAxisRoll, polRoll, depolRoll = calcLines(raw, d, r)
    polDots.set_data(xAxisDots, polDotVals)   
    depolDots.set_data(xAxisDots, depolDotVals)
    polLine.set_data(xAxisRoll, polRoll)
    depolLine.set_data(xAxisRoll, depolRoll)
    ax.set_ylim(bottom=np.min(np.minimum(polDotVals, depolDotVals)), top=np.max(np.maximum(polDotVals, depolDotVals)))
    fig.canvas.draw_idle()      #re-draws the plot

diffSlider.on_changed(update)
rollSlider.on_changed(update)   #calls update function if slider is changed

plt.show()



