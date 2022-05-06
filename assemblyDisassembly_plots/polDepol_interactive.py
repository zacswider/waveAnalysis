import sys
import numpy as np
import pandas as pd
import skimage.io as skio  
import matplotlib.pyplot as plt    
from matplotlib.widgets import Slider, Button
from skimage.transform import downscale_local_mean as downscale                     

# enter file path here
rawFilePath = ' '

# enter save path here
savePath = ' '

# lazy quality control
raw_img = skio.imread(rawFilePath).astype('float64') # should be array of shape (frames, y, x)
if raw_img.ndim > 3:
    print("was NOT processed. Are you sure you have a standard sized image with one or two channels saved in standard `tzcyx` order?")
    sys.exit()

print(f'starting with image of shape {raw_img.shape}')

# if desired, enter a scale factor > 1 to accelerate computation
scaleFactor = 4
downsample_dims = (1, scaleFactor, scaleFactor)
scaled_img = downscale(raw_img, downsample_dims)
print(f'processing image of shape {scaled_img.shape}')

# starting plot parameters
diffNumber = 3
windowSize = 3
plotData = True

# difference calculations
def calcLines(im, diffNum, window):
    diff = np.subtract(im[diffNum:], im[:-diffNum])
    polZAxis =  np.nansum(np.where(diff>0, diff, np.nan), axis=(1,2))           
    depolZAxis = np.abs(np.nansum(np.where(diff<0, diff, np.nan), axis=(1,2)))  
    numPoints = int(im.shape[0]-diffNum)
    diffXAxis = np.linspace(1, numPoints, numPoints)
    rollPol = np.convolve(polZAxis, np.ones(window), 'valid') / window      #returns rolling average array
    rollDepol = np.convolve(depolZAxis, np.ones(window), 'valid') / window      #returns rolling average array
    numPointsRoll = int(im.shape[0]-diffNum-window//2)
    rollXAxis = np.linspace((1+window//2), (numPointsRoll-window//2), (numPointsRoll-window//2), dtype=int)
    return(diffXAxis, polZAxis, depolZAxis, rollXAxis, rollPol, rollDepol)

# re-draw the plot
def update(val):
    d = int(diffSlider.val)
    r = int(rollSlider.val)
    xAxisDots, polDotVals, depolDotVals, xAxisRoll, polRoll, depolRoll = calcLines(scaled_img, d, r)
    polDots.set_data(xAxisDots, polDotVals)   
    depolDots.set_data(xAxisDots, depolDotVals)
    polLine.set_data(xAxisRoll, polRoll)
    depolLine.set_data(xAxisRoll, depolRoll)
    ax.set_ylim(bottom=np.min(np.minimum(polDotVals, depolDotVals)), top=np.max(np.maximum(polDotVals, depolDotVals)))
    fig.canvas.draw_idle()

# save the plot values on button push
def save_plot_vals(val):
    df1 = pd.DataFrame({"xAxisDots":xAxisDots, "polDotVals":polDotVals, "depolDotVals":depolDotVals})
    df2 = pd.DataFrame({"xAxisRoll":xAxisRoll, "polRoll":polRoll, "depolRoll":depolRoll})
    df = pd.concat([df1, df2], ignore_index=False, axis=1)
    df.to_csv(savePath + "plot.csv")

# make fig and ax objects
fig, ax = plt.subplots(figsize=(7, 5))
fig.subplots_adjust(bottom=0.2, top=0.75)

# make a decorate the sliders
diffAx = fig.add_axes([0.25, 0.85, 0.4, 0.05]) 
rollAx = fig.add_axes([0.25, 0.92, 0.4, 0.05])
diffSlider = Slider(ax=diffAx, label='Frames to difference ', valmin=1, valmax=15, valinit=diffNumber, valfmt=' %1.1f Frames', valstep = 1, facecolor='#cc7000')
rollSlider = Slider(ax=rollAx, label='Frames to average ', valmin=1, valmax=15, valinit=windowSize, valfmt='%i Frames', valstep = 2, facecolor='#cc7000')

# make a decorate the save button
ax_button = fig.add_axes([0.82, 0.85, 0.08, 0.05])
grid_button = Button(ax_button, 'Save', color='#cc7000', hovercolor='grey')

# decorate the starting plot
ax.set_ylabel('relative assembly and disassembly')
ax.set_xlabel('time (frames)')
xAxisDots, polDotVals, depolDotVals, xAxisRoll, polRoll, depolRoll = calcLines(scaled_img, diffNumber, windowSize)
polDots, = ax.plot(xAxisDots, polDotVals, color='deepskyblue', marker='o', linestyle="", markersize=2, alpha=0.25)
depolDots, = ax.plot(xAxisDots, depolDotVals, color='darkorange', marker='o', linestyle="", markersize=2, alpha=0.25)
polLine, = ax.plot(xAxisRoll, polRoll, color='deepskyblue', label='total net F-actin assembly')
depolLine, = ax.plot(xAxisRoll, depolRoll, color='darkorange', label='total net F-actin disassembly')
ax.legend(loc='upper right', fontsize='small', frameon=False, ncol=1)

# slider move and button push calls
diffSlider.on_changed(update)
rollSlider.on_changed(update)
grid_button.on_clicked(save_plot_vals)

plt.show()



