import numpy as np
import pandas as pd  
import seaborn as sns
import skimage.io as skio  
import scipy.signal as sig                        
import matplotlib.pyplot as plt    
from matplotlib.widgets import Slider
from tkinter.filedialog import askdirectory     
from matplotlib.animation import FuncAnimation   

rawFilePath = "/Users/bementmbp/Desktop/polDepolTesting/200630_Live_SFC_Aegg_GFP-Utr_2mCh-EMTB_E06-T01_Max-UtrCrop_260-620.tif"
raw = skio.imread(rawFilePath).astype('float64') #array of shape (frames, y, x)
diffNumber = 5
windowSize = 5

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
    return(diffXAxis, polZAxis, depolZAxis, rollXAxis, rollPol, rollDepol)

fig = plt.figure(figsize=(7, 5))        #figure object
ax = fig.add_subplot(111)               #Create main axis; 111=row,column,position. Not strictly necessary with only one subplot
fig.subplots_adjust(bottom=0.2, top=0.75)   #position as a fraction of the figure width
diffAx = fig.add_axes([0.3, 0.85, 0.4, 0.05])    #rectangle of size [x0, y0, width, height]
rollAx = fig.add_axes([0.3, 0.92, 0.4, 0.05])
rollValues = np.linspace(1,15,8)
diffValues = np.linspace(1,15,15)
diffSlider = Slider(ax=diffAx, label='Frames to difference ', valmin=0, valmax=15, valinit=5,valfmt=' %1.1f Frames', valstep=diffValues, facecolor='#cc7000')
rollSlider = Slider(ax=rollAx, label='Frames to average ', valmin=1, valmax=15, valinit=5, valfmt='%i Frames', valstep=rollValues, facecolor='#cc7000')
ax.set_ylabel('relative assembly and disassembly')
ax.set_xlabel('time (frames)')

xAxisDots, polDotVals, depolDotVals, xAxisRoll, polRoll, depolRoll = calcLines(raw, diffNumber, windowSize)
polDots, = ax.plot(xAxisDots, polDotVals, color='deepskyblue', marker='o', linestyle="", markersize=2, alpha=0.25)
depolDots, = ax.plot(xAxisDots, depolDotVals, color='darkorange', marker='o', linestyle="", markersize=2, alpha=0.25)
polLine, = ax.plot(xAxisRoll, polRoll, color='deepskyblue', label='recent F-actin assembly')
depolLine, = ax.plot(xAxisRoll, depolRoll, color='darkorange', label='recent F-actin disassembly')
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








"""

rawFilePath = "/Users/bementmbp/Desktop/polDepolTesting/200630_Live_SFC_Aegg_GFP-Utr_2mCh-EMTB_E06-T01_Max-UtrCrop_260-620.tif"
raw = skio.imread(rawFilePath).astype('float64') #array of shape (frames, y, x)
diffNumber = 5
windowSize = 5

def calcPolZAxis(rawData, diffNum):
    diff = np.subtract(rawData[diffNum:], rawData[:-diffNum])
    return np.nanmean(np.where(diff>0, diff, np.nan), axis=(1,2))

def calcDepolZAxis(rawData, diffNum):
    diff = np.subtract(rawData[diffNum:], rawData[:-diffNum])
    return np.abs(np.nanmean(np.where(diff<0, diff, np.nan), axis=(1,2)))

def calcDiffX(rawData, diffNum):
    numPoints = int(rawData.shape[0]-diffNum)
    return np.linspace(1, numPoints, numPoints)

def movingAvgPol(rawData, diffNum, window):                                #accepts a 1D array (x) and window of length to be averaged (w)
    diff = np.subtract(rawData[diffNum:], rawData[:-diffNum])
    diffZ =  np.nanmean(np.where(diff>0, diff, np.nan), axis=(1,2))
    return np.convolve(diffZ, np.ones(window), 'valid') / window      #returns rolling average array

def movingAvgDepol(rawData, diffNum, window):                                #accepts a 1D array (x) and window of length to be averaged (w)
    diff = np.subtract(rawData[diffNum:], rawData[:-diffNum])
    diffZ =  abs(np.nanmean(np.where(diff<0, diff, np.nan), axis=(1,2)))
    return np.convolve(diffZ, np.ones(window), 'valid') / window      #returns rolling average array

def calcMovingAxis(rawData, diffNum, window):
    numPoints = int(rawData.shape[0]-diffNum-window//2)
    return np.linspace((1+window//2), (numPoints-window//2), (numPoints-window//2))

fig = plt.figure(figsize=(6, 4))        #figure object
ax = fig.add_subplot(111)               #Create main axis; 111=row,column,position. Not strictly necessary with only one subplot
fig.subplots_adjust(bottom=0.2, top=0.75)   #position as a fraction of the figure width
diffAx = fig.add_axes([0.3, 0.85, 0.4, 0.05])    #rectangle of size [x0, y0, width, height]
rollAx = fig.add_axes([0.3, 0.92, 0.4, 0.05])
rollValues = np.linspace(1,15,8)
diffValues = np.linspace(1,15,15)
diffSlider = Slider(ax=diffAx, label='Frames to difference ', valmin=0, valmax=15, valinit=5,valfmt=' %1.1f Frames', valstep=diffValues, facecolor='#cc7000')
rollSlider = Slider(ax=rollAx, label='Frames to average ', valmin=1, valmax=15, valinit=5, valfmt='%i Frames', valstep=rollValues, facecolor='#cc7000')
ax.set_ylabel('relative assembly and disassembly')
ax.set_xlabel('time (frames)')



polDots, = ax.plot(calcDiffX(raw, diffNumber), 
                   calcPolZAxis(raw, diffNumber), 
                   color='deepskyblue', marker='o', linestyle="", markersize=2, alpha=0.25)
depolDots, = ax.plot(calcDiffX(raw, diffNumber), 
                     calcDepolZAxis(raw, diffNumber), 
                     color='darkorange', marker='o', linestyle="", markersize=2, alpha=0.25)
polLine, = ax.plot(calcMovingAxis(raw, diffNumber, windowSize), 
                   movingAvgPol(raw, diffNumber, windowSize), color='deepskyblue')
depolLine, = ax.plot(calcMovingAxis(raw, diffNumber, windowSize), 
                   movingAvgDepol(raw, diffNumber, windowSize), color='darkorange')

def update(val):
    d = int(diffSlider.val)
    r = int(rollSlider.val)
    polDots.set_data(calcDiffX(raw, d), calcPolZAxis(raw, d))    #sets new data for the line object, overriding old data.
    depolDots.set_data(calcDiffX(raw, d), calcDepolZAxis(raw, d))
    polLine.set_data(calcMovingAxis(raw, d, r), movingAvgPol(raw, d, r))
    depolLine.set_data(calcMovingAxis(raw, d, r), movingAvgDepol(raw, d, r))
    ax.set_ylim(bottom = np.min(np.minimum(calcPolZAxis(raw, d), calcDepolZAxis(raw, d))), top=np.max(np.maximum(calcPolZAxis(raw, d), calcDepolZAxis(raw, d))))
    fig.canvas.draw_idle()      #re-draws the plot

diffSlider.on_changed(update)
rollSlider.on_changed(update)   #calls update function if slider is changed

plt.show()




"""



"""
more negatives values means that it's averaging less zeros which is going to amplify the signal, moreover the 
less positive values means that is's avering MORE zeros which is going to dampen the signal. By including zeros
in the math I'm not exactly creating an artifact per se, but I'm artificially amplifying the contrast. 

def movingAverage(x, w):                                #accepts a 1D array (x) and window of length to be averaged (w)
    return np.convolve(x, np.ones(w), 'valid') / w      #returns rolling average array

def calcDiff(rawData, diffNum):
    return np.subtract(rawData[diffNum:], rawData[:-diffNum])

rawFilePath = "/Users/bementmbp/Desktop/polDepolTesting/200630_Live_SFC_Aegg_GFP-Utr_2mCh-EMTB_E06-T01_Max-UtrCrop_260-620.tif"
raw = skio.imread(rawFilePath).astype('float64') #array of shape (frames, y, x)
diffNum = 6
diff = calcDiff(raw, diffNum)
pol = diff.clip(min=0)                          #clips all negatives to zeros
depol = diff.clip(max=0)                        #clips all positives to zeros
polZAxis = np.mean(pol, axis=(1,2))                     #the "z-axis profile", ie mean of each frame as 
depolZAxis = np.abs(np.mean(depol, axis=(1,2)))                #the "z-axis profile", ie mean of each frame as 
plt.plot(polZAxis)
plt.plot(depolZAxis)
plt.show()
"""

"""
polFilePath = "/Users/bementmbp/Desktop/polDepolTesting/200630_Live_SFC_Aegg_GFP-Utr_2mCh-EMTB_E06-T01_Max-UtrCrop_260-620_PolDiff6.tif"      
depolFilePath = "/Users/bementmbp/Desktop/polDepolTesting/200630_Live_SFC_Aegg_GFP-Utr_2mCh-EMTB_E06-T01_Max-UtrCrop_260-620_DepolDiff6.tif" 
pol = skio.imread(polFilePath)
depol = skio.imread(depolFilePath)

def movingAverage(x, w):                                #accepts a 1D array (x) and window of length to be averaged (w)
    return np.convolve(x, np.ones(w), 'valid') / w      #returns rolling average array

windowSize = 5
polZAxis = np.mean(pol, axis=(1,2))                     #the "z-axis profile", ie mean of each frame as 
depolZAxis = np.mean(depol, axis=(1,2))                 #the "z-axis profile", ie mean of each frame as 
movingPol = movingAverage(polZAxis, windowSize)         #moving average of 5 frames of pol
movingDepol = movingAverage(depolZAxis, windowSize)     #moving average of 5 frames of depol
startFrameRaw = 1                                       #starting frame (x-axis) for the raw data
startFrameRoll = startFrameRaw + windowSize//2          #starting frame (x-axis) for the rolling avg     

xAxis=np.linspace(startFrameRaw,pol.shape[0],pol.shape[0])                      #makes x-axis for raw data
movingX=np.linspace(startFrameRoll, movingPol.shape[0], movingPol.shape[0])     #makes x-axis for rolling avg

fig = plt.figure()#figsize=(6, 4))        #figure object
ax = fig.add_subplot(111)               #Create main axis; 111=row,column,position. Not strictly necessary with only one subplot
#fig.subplots_adjust(bottom=0.2, top=0.75)   #position as a fraction of the figure width
##ax.set_ylabel('relative assembly and disassembly')
#ax.set_xlabel('time (frames)')
ax.plot(xAxis, polZAxis)#, color='deepskyblue', marker='o', linestyle="", markersize=2, alpha=0.25)
ax.plot(xAxis, depolZAxis)#, color='darkorange', marker='o', linestyle="", markersize=2, alpha=0.25)
#ax.plot(movingX, movingPol, color='deepskyblue')
#ax.plot(movingX, movingDepol, color='darkorange')

plt.show()

"""

