import numpy as np
import pandas as pd  
import seaborn as sns
import skimage.io as skio  
from skimage.transform import downscale_local_mean as downscale
import scipy.signal as sig                        
import matplotlib.pyplot as plt    
from matplotlib.widgets import Slider
from tkinter.filedialog import askdirectory     
from matplotlib.animation import FuncAnimation   

rawFilePath = "/Users/bementmbp/Desktop/BementLab/2_Projects/23_DevPaper/Figures/Figure4E/190219_Live_Flvw_Emb_Utr647_E02-T01_2-323_bleachCorr_CropUtr_40-250.tif"
foo = skio.imread(rawFilePath).astype('float64') #array of shape (frames, y, x)
print(foo.shape)
scaleFactor = 2
factorArray = (1, scaleFactor, scaleFactor)
raw = downscale(foo, factorArray)
print(raw.shape)
diffNumber = 3
windowSize = 3
plotData = True
savePath = "/Users/bementmbp/Desktop/"

def calcLines(rawData, diffNum, window):
    diff = np.subtract(rawData[diffNum:], rawData[:-diffNum])
    polZAxis =  np.nansum(np.where(diff>0, diff, np.nan), axis=(1,2))           #this is now TOTAL disassembly
    depolZAxis = np.abs(np.nansum(np.where(diff<0, diff, np.nan), axis=(1,2)))  #this is now the RATE of disassembly
    numPoints = int(rawData.shape[0]-diffNum)
    diffXAxis = np.linspace(1, numPoints, numPoints)
    rollPol = np.convolve(polZAxis, np.ones(window), 'valid') / window      #returns rolling average array
    rollDepol = np.convolve(depolZAxis, np.ones(window), 'valid') / window      #returns rolling average array
    numPointsRoll = int(rawData.shape[0]-diffNum-window//2)
    rollXAxis = np.linspace((1+window//2), (numPointsRoll-window//2), (numPointsRoll-window//2), dtype=int)
    return(diffXAxis, polZAxis, depolZAxis, rollXAxis, rollPol, rollDepol)

fig = plt.figure(figsize=(7, 5))        #figure object
ax = fig.add_subplot(111)               #Create main axis; 111=row,column,position. Not strictly necessary with only one subplot
fig.subplots_adjust(bottom=0.2, top=0.75)   #position as a fraction of the figure width
diffAx = fig.add_axes([0.3, 0.85, 0.4, 0.05])    #rectangle of size [x0, y0, width, height]
rollAx = fig.add_axes([0.3, 0.92, 0.4, 0.05])
rollValues = np.array([1,3,5,7,9,11,13,15])
diffValues = np.linspace(1,15,15)
try:
    diffSlider = Slider(ax=diffAx, label='Frames to difference ', valmin=1, valmax=15, valinit=diffNumber, valfmt=' %1.1f Frames', valstep=diffValues, facecolor='#cc7000')
except ValueError:
    diffSlider = Slider(ax=diffAx, label='Frames to difference ', valmin=1, valmax=15, valinit=diffNumber, valfmt=' %1.1f Frames', valstep=diffValues.all(), facecolor='#cc7000')
try:
    rollSlider = Slider(ax=rollAx, label='Frames to average ', valmin=1, valmax=15, valinit=windowSize, valfmt='%i Frames', valstep=rollValues, facecolor='#cc7000')
except ValueError:
    rollSlider = Slider(ax=rollAx, label='Frames to average ', valmin=1, valmax=15, valinit=windowSize, valfmt='%i Frames', valstep=rollValues.all(), facecolor='#cc7000')
ax.set_ylabel('relative assembly and disassembly')
ax.set_xlabel('time (frames)')

xAxisDots, polDotVals, depolDotVals, xAxisRoll, polRoll, depolRoll, rawMean = calcLines(raw, diffNumber, windowSize)
polDots, = ax.plot(xAxisDots, polDotVals, color='deepskyblue', marker='o', linestyle="", markersize=2, alpha=0.25)
depolDots, = ax.plot(xAxisDots, depolDotVals, color='darkorange', marker='o', linestyle="", markersize=2, alpha=0.25)
polLine, = ax.plot(xAxisRoll, polRoll, color='deepskyblue', label='recent F-actin assembly')
depolLine, = ax.plot(xAxisRoll, depolRoll, color='darkorange', label='recent F-actin disassembly')
#ax.plot(rawMean)
ax.legend(loc='upper right', fontsize='small', frameon=False, ncol=1)

if plotData == True:
    df1 = pd.DataFrame({"xAxisDots":xAxisDots, "polDotVals":polDotVals, "depolDotVals":depolDotVals})
    df2 = pd.DataFrame({"xAxisRoll":xAxisRoll, "polRoll":polRoll, "depolRoll":depolRoll})
    df = pd.concat([df1, df2], ignore_index=False, axis=1)
    df.to_csv(savePath + "plot.csv")

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



