import os                                       
import sys                                      
import math
import pathlib   
import fnmatch
import datetime                               
import numpy as np
import pandas as pd  
import seaborn as sns
import tkinter as tk
from tkinter import Tk
from tkinter import ttk
import skimage.io as skio  
import scipy.signal as sig
from genericpath import exists            
import matplotlib.pyplot as plt    
from tkinter.filedialog import askdirectory   
import timeit


np.seterr(divide='ignore', invalid='ignore')

'''*** Start GUI Window ***'''

#initiates Tk window
root = tk.Tk()
root.title('Select your options')
root.geometry('500x250')

#sets number of columns in the main window
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)

#defining variable types for the different widget fields
boxSizeVar = tk.IntVar()            #variable for box grid size
boxSizeVar.set(20)                  #set default value 
plotIndividualACFsVar = tk.BooleanVar()     #variable for plotting individual ACFs
plotIndividualCCFsVar = tk.BooleanVar()     #variable for plotting individual CCFs
plotIndividualPeaksVar = tk.BooleanVar()    #variable for plotting individual peaks
acfPeakPromVar = tk.DoubleVar()             #variable for peak prominance threshold   
acfPeakPromVar.set(0.1)                     #set default value
groupNamesVar = tk.StringVar()   #variable for group names list
folderPath = tk.StringVar()      #variable for path to images

#function for getting path to user's directory
def getFolderPath():
    folderSelected = askdirectory()
    folderPath.set(folderSelected)

#function for hitting cancel button or quitting
def on_quit(): 
    root.destroy() #destroys window
    sys.exit("You opted to cancel the script!")

#function for hitting start button
def on_start(): 
        root.destroy() #destroys window
    

'''widget creation'''
#file path selection widget
fileEntry = ttk.Entry(root, textvariable=folderPath)
fileEntry.grid(column=0, row=0, padx=10, sticky='E')
browseButton = ttk.Button(root, text= 'Select source directory', command=getFolderPath)
browseButton.grid(column=1, row=0, sticky='W')

#boxSize entry widget
boxSizeBox = ttk.Entry(root, width = 3, textvariable=boxSizeVar) #creates box widget
boxSizeBox.grid(column=0, row=1, padx=10, sticky='E') #places widget in frame
boxSizeBox.focus()      #focuses cursor in box
boxSizeBox.icursor(2)   #positions cursor after default input characters
ttk.Label(root, text='Enter grid box size (px)').grid(column=1, row=1, columnspan=2, padx=10, sticky='W') #create label text

#create acfpeakprom entry widget
ttk.Entry(root, width = 3, textvariable=acfPeakPromVar).grid(column=0, row=2, padx=10, sticky='E') #create the widget
ttk.Label(root, text='Enter ACF peak prominence threshold').grid(column=1, row=2, padx=10, sticky='W') #create label text

#create groupNames entry widget
ttk.Entry(root,textvariable=groupNamesVar).grid(column=0, row=3, padx=10, sticky='E') #create the widget
ttk.Label(root, text='Enter group names separated by commas').grid(column=1, row=3, padx=10, sticky='W') #create label text

#create checkbox widgets and labels
ttk.Checkbutton(root, variable=plotIndividualACFsVar).grid(column=0, row=5, sticky='E', padx=15)
ttk.Label(root, text='Plot individual ACFs').grid(column=1, row=5, columnspan=2, padx=10, sticky='W') #plot individual ACFs
ttk.Checkbutton(root, variable=plotIndividualCCFsVar).grid(column=0, row=6, sticky='E', padx=15) #plot individual CCFs
ttk.Label(root, text='Plot individual CCFs').grid(column=1, row=6, columnspan=2, padx=10, sticky='W')

ttk.Checkbutton(root, variable=plotIndividualPeaksVar).grid(column=0, row=7, sticky='E', padx=15) #plot individual peaks
ttk.Label(root, text='Plot individual peaks').grid(column=1, row=7, columnspan=2, padx=10, sticky='W')

#Creates the 'Start Analysis' button
startButton = ttk.Button(root, text='Start Analysis', command=on_start) #creates the button and bind it to close the window when clicked
startButton.grid(column=1, row=9, pady=10, sticky='W') #place it in the tk window

#Creates the 'Cancel' button
cancelButton = ttk.Button(root, text='Cancel', command=on_quit) #creates the button and bind it to on_quit function
cancelButton.grid(column=0, row=9, pady=10, sticky='E') #place it in the tk window

root.protocol("WM_DELETE_WINDOW", on_quit) #calls on_quit if the root window is x'd out.
root.mainloop() #run the script

#get the values stored in the widget
boxSizeInPx = boxSizeVar.get()
plotIndividualACFs= plotIndividualACFsVar.get()
plotIndividualCCFs = plotIndividualCCFsVar.get()
plotIndividualPeaks = plotIndividualPeaksVar.get()
acfPeakProm = acfPeakPromVar.get()
groupNames = groupNamesVar.get()
groupNames = [x.strip() for x in groupNames.split(',')] #list of group names. splits string input by commans and removes spaces
baseDirectory = folderPath.get() 

#make dictionary of parameters for log file use
logParams = {
    "Box Size(px)" : boxSizeInPx,
    "Base Directory" : baseDirectory,
    "ACF Peak Prominence" : acfPeakProm,
    "Group Names" : groupNames,
    "Plot Individual ACFs" : plotIndividualACFs,
    "Plot Individual CCFs" : plotIndividualCCFs,
    }

errors = []
if acfPeakProm > 1 :
    errors.append("The ACF peak prominence can not be greater than 1, set 'ACF peak prominence threshold' to a value between 0 and 1. More realistically, a value between 0 and 0.5")
if len(baseDirectory) < 1 :
    errors.append("You didn't enter a directory to analyze")

if len(errors) >= 1 :
    print("Error Log:")
    for count, error in enumerate(errors):
        print(count,":", error)
    sys.exit("Please fix errors and try again.") 

'''*** End GUI Window ***'''

'''*** Start Processing Functions ***'''
def findWorkspace(directory):                                                       #accepts a starting directory and a prompt for the GUI
    Tk().withdraw()
    filelist = [fname for fname in os.listdir(directory) if fname.endswith('.tif')]   #Makes a list of file names that end with .tif
    return(filelist)                                                       #returns the folder path and list of file names

def setGroups(groupNames, nameWithoutExtension):
    for group in groupNames:
        if group in nameWithoutExtension:
            return(group)

def smoothWithSavgol(signal, windowSize, polynomial):                       #accepts a signal array (or list), number of values to match, and polynomial number.
    smoothedSignal = sig.savgol_filter(signal, windowSize, polynomial)      #smooths the input signal...
    return smoothedSignal                                                   #...and returns it

def findBoxMeans(imageArray, boxSize):              #accepts an image array as a parameter, as well as the desired box size
    depth = imageArray.shape[0]                     #number of frames
    yDims = imageArray.shape[1]                     #number of pixels on y-axis
    xDims = imageArray.shape[2]                     #number of pixels on x-axis
    yBoxes = yDims // boxSize                       #returns int result of floor division; number of boxes on the y axis
    xBoxes = xDims // boxSize                       #returns int result of floor division; number of boxes on the x axis
    growingArray = np.zeros((xBoxes, yBoxes, depth))#makes a starting array of 64 bit zeros that can be modified later. shape = (num x boxes, num y boxes, depth of imageStack)

    for x in range(xBoxes):                         #iterates through the number of boxes on the x-axis
        for y in range (yBoxes):                    #iterates through the number of boxes on the y-axis
            boxMean = np.array([np.mean(imageArray[:,(y*boxSize):(y*boxSize+boxSize),(x*boxSize):(x*boxSize+boxSize)], (1,2))])  
            #creates a 2d array of shape (depth, 1) containing the mean values of the px within the box for each slices
            growingArray[x][y] = boxMean            #reassigns zero values at this position to the box mean values
    growingArray = np.reshape(growingArray, (growingArray.shape[0]*growingArray.shape[1], growingArray.shape[2])) #reshapes to a 2d instead of 3d array
    return(growingArray)                            #returns ndarray of shape (number of boxes, number of frames)

def printBoxACF(signal, acor, boxNum, directory, channel="", delay=None):   #Accepts a signal and an autocorrelation to plot
    acfSavePath = directory / ("boxGraphs") / (channel + "ACF_Plots")       #Specifies subfolder path
    acfSavePath.mkdir(exist_ok=True, parents=True)                          #Makes the subfolder
    xAxis = np.arange(signal.shape[0])                                      #x-axis for the signal plot
    lags = np.arange(-signal.shape[0] + 1, signal.shape[0])                 #x-axis for the autocorrelation
    fig, axs = plt.subplots(nrows=2)                                        #subplot with two rows
    fig.subplots_adjust(hspace=0.4)                                         #set white space between plots
    ax = axs[0]                                                             #start plotting the first row
    ax.plot(xAxis, signal)                                                  #plots the signal against its x-axis
    ax.set_ylabel('Mean box px value')                                      #sets y-axis label
    ax.set_xlabel('Time (frames)')                                          #sets x-axis label

    if delay == None:                                                       #if no delay variable is passed to the function
        ax = axs[1]                                                         #start plotting the second row
        ax.plot(lags, acor)                                                 #plots the acor against it x-axis   
        ax.set_ylabel('Auto-correlation')                                   #sets y-axis label
        ax.set_xlabel("Periodic signal not detected")                       #sets x-axis label; assumes no periodic signal detected if no delay passed
        boxName = acfSavePath / ("boxNo" + str(boxNum) + ".png")            #names the figure                                 
        plt.savefig(boxName, dpi=75, )                                      #saves the figure 
        plt.close(fig)                                                      #clears the figure

    else:                                                                   #if a delay IS passed to the function
        ax = axs[1]                                                         #start plotting the second row
        ax.plot(lags, acor)                                                 #plots the acor against it x-axis       
        ax.set_ylabel('Auto-correlation')                                   #sets y-axis label
        ax.set_xlabel("Period is " + str(delay) + " frames")                #sets x-axis label specifying the period passed to the function
        plt.axvline(x=delay, alpha = 0.5, c = 'red', linestyle = '--')      #adds a vertical line identifying the chosen peak 
        plt.axvline(x=-delay, alpha = 0.5, c = 'red', linestyle = '--')     #same as above, but in negative space
        boxName = acfSavePath / ("boxNo" + str(boxNum) + ".png")            #names the figure
        plt.savefig(boxName, dpi=75)                                        #saves the figure
        plt.close(fig)                                                      #clears the figure

def printBoxCCF(signal1, signal2, ccor, boxNum, directory, shift):   #Accepts two signals and a crosscorrelation to plot
    ccfSavePath = directory / ("boxGraphs") / ("CCF_Plots")                 #Specifies subfolder path
    ccfSavePath.mkdir(exist_ok=True, parents=True)                          #Makes the subfolder
    assert len(signal1) == len(signal2), "signals must be the same size"    #user feedback
    xAxis = np.arange(signal1.shape[0])                                     #x-axis for the signal plot
    lags = np.arange(-signal1.shape[0] + 1, signal1.shape[0])               #x-axis for the autocorrelation
    signal1 = (signal1-np.min(signal1))/(np.max(signal1)-np.min(signal1))   #normalizes signal1 to 0-1
    signal2 = (signal2-np.min(signal2))/(np.max(signal2)-np.min(signal2))   #normalizes signal2 to 0-1
    fig, axs = plt.subplots(nrows=2)                                        #subplot with two rows
    fig.subplots_adjust(hspace=0.4)                                         #set white space between plots
    ax = axs[0]                                                             #start plotting the first row
    ax.plot(xAxis, signal1, 'tab:orange', label='Ch1'),                     #plots signal1 against its x-axis
    ax.plot(xAxis, signal2, 'tab:cyan', label='Ch2'),                       #plots signal2 against its x-axis
    ax.set_ylabel('norm box px value (AU)')                                 #sets y-axis label
    ax.set_xlabel('Time (frames)')                                          #sets x-axis label
    ax.legend(loc='upper right', fontsize='small', ncol=1)                  #places the fig legend
    ax = axs[1]                                                             #start plotting the second row
    ax.plot(lags, ccor)                                                     #plots the ccor against it x-axis       
    ax.set_ylabel('Cross-correlation')                                      #sets y-axis label
    if shift < 0:
        ax.set_xlabel("Ch1 leads by " + str(abs(shift)) + " frames")        #sets x-axis label specifying that Ch1 leads
    elif shift > 0:
        ax.set_xlabel("Ch2 leads by " + str(abs(shift)) + " frames")        #sets x-axis label specifying that Ch2 leads    
    else:
        ax.set_xlabel("There is no detectable shift between signals")       #sets x-axis label specifying that neither channel leads
    plt.axvline(x=shift, alpha = 0.5, c = 'red', linestyle = '--')          #adds a vertical line identifying the chosen peak 
    boxName = ccfSavePath / ("boxNo" + str(boxNum) + ".png")                #names the figure
    plt.savefig(boxName, dpi=75)                                            #saves the figure
    plt.close(fig)                                                          #clears the figure

def findACF(signal, directory, boxNumber, channel):             #accepts a single array (one) channels), which will be correlated to itself.
    npts = signal.shape[0]                                      #number of points is the depth of the 0 axis, which is the number of frames in the image
    acov = np.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')    #compute full autocorrelation
    acor = acov / (npts * signal.std() ** 2)                    #normalizes the crosscorr from -1 to +1      
    #acor = smoothWithSavgol(acor, windowSize=3, polynomial=1)  #smooths the ACF to better eliminate noisy peaks. Keeping this for troubleshooting purposes
    peaks, dict = sig.find_peaks(acor, prominence=acfPeakProm)  #ndarray with location of local maxima using scipy.signal.find_peaks
    peaksDiff = abs(peaks - acor.shape[0]//2)                   #ndarray with the absolute difference between each peak and the middle value of the ccor array

    try:
        delay = np.min(peaksDiff[np.nonzero(peaksDiff)])        #numpy.int64 reporting difference between the first peak and zero
        if plotIndividualACFs == True:                          #user set at the top of the script
            print("plotting ACF from box #" + str(boxNumber) + " for " + channel)            #terminal feedback for script progression
            printBoxACF(signal, acor, boxNumber, directory, channel, delay)         #calls the print box acf function
    except ValueError:                                          #if no suitable peak is identified...
        if plotIndividualACFs == True:                          #user set at the top of the script
            print("plotting ACF from box #" + str(boxNumber) + " for " + channel)            #terminal feedback for script progression
            printBoxACF(signal, acor, boxNumber, directory, channel, delay=None)    #calls the print box acf function    
        zeroArray = np.full((npts*2-1), np.nan)                 #array of nan values w/ shape (x,) where x = the number of points in the ccor
        return(zeroArray, np.nan)                               #returns an empty acor cure and delay; 
                                                                #ensures that a box with no detectable period will not be averaged into the pop ACF or pop stats    
    return(acor, delay)                                         #returns the acor curve and delay. 

def findCCF(signal1, signal2, directory, boxNumber):        #accepts a single array (one) channels), which will be correlated to itself.
    assert len(signal1) == len(signal2), "input arrays must be the same size"
    npts = signal1.shape[0]                                 #number of points is the depth of the 0 axis, which is the number of frames in the image
    ccov = np.correlate(signal1 - signal1.mean(), signal2 - signal2.mean(), mode='full')    #compute full autocorrelation
    ccor = ccov / (npts * signal1.std() * signal2.std())    #normalizes the crosscorr from -1 to +1      
    peaks, dict = sig.find_peaks(ccor)                      #ndarray with location of local maxima using scipy.signal.find_peaks
    peaksDiff = abs(peaks - ccor.shape[0]//2)               #ndarray with the absolute difference between each peak and the middle value of the ccor array
    delay = np.argmin(peaksDiff)                            #numpy.int64 reporting difference between the first peak and zero
    delayIndex = peaks[delay]                               #index of the peak
    actualShift = delayIndex - ccor.shape[0]//2             #actual shift value of first maxima minus middle of ccor
    
    if plotIndividualCCFs == True:                          #user set at the top of the script
        print("plotting CCF from box #" + str(boxNumber))   #terminal feedback for script progression
        printBoxCCF(signal1, signal2, ccor, boxNumber, directory, actualShift)  #calls plot ccf function

    return(ccor, actualShift)                               #returns the ccor and the shift between signals

def printBoxPeaks(raw, smoothed, smoothPeaks, heights, leftIndex, rightIndex, proms, directory, boxNumber, channel):
    peaksSavePath = directory / ("boxGraphs") / (channel+"Peak_Plots")  #specifies path
    peaksSavePath.mkdir(exist_ok=True, parents=True)                    #makes path

    x = np.arange(raw.shape[0])                                     #x-axis
    fig, axs = plt.subplots(nrows=2)                                #subplot with two rows
    fig.subplots_adjust(hspace=0.4)                                 #set white space between plots
    ax = axs[0]                                                     #start plotting first row
    ax.plot(x,raw, color='tab:blue', label='raw ' + channel)        #plot raw trace
    ax.plot(x,smoothed, color='tab:orange', label='smoothed')       #plot smoothed trace
    ax.legend(loc='upper right', fontsize='small', ncol=1)          #set legend
    ax.set_ylabel('Mean box px value (AU)')                         #y-axis label
    ax.set_xlabel('Time (frames)')                                  #x-axis label
    ax=axs[1]                                                       #start plotting second row
    ax.plot(x,smoothed, color='tab:orange', label='smoothed')       #plot smoothed trace
    for i in range(smoothPeaks.shape[0]):                           #plot peaks
        ax.hlines(heights[i], leftIndex[i], rightIndex[i], color='tab:blue', alpha = 1, linestyle = '-')
        ax.vlines(smoothPeaks[i], smoothed[smoothPeaks[i]]-proms[i], smoothed[smoothPeaks[i]], color='tab:purple', alpha = 1, linestyle = '-')
    ax.hlines(heights[0], leftIndex[0], rightIndex[0], color='tab:blue', alpha = 1, linestyle = '-', label='FWHM')  #plot peaks again to allow specifying legend
    ax.vlines(smoothPeaks[0], smoothed[smoothPeaks[0]]-proms[0], smoothed[smoothPeaks[0]], color='tab:purple', alpha = 1, linestyle = '-', label='Peak amplitude')
    ax.legend(loc='upper right', fontsize='small', ncol=1)          #set legend
    ax.set_ylabel('Mean box px value')                              #y-axis label
    ax.set_xlabel('Time (frames)')                                  #x-axis label
    boxName = peaksSavePath / (channel + "_boxNo" + str(boxNumber) + ".png")    #names the figure
    plt.savefig(boxName, dpi=74)                                                #saves the figure
    plt.close(fig)                                                              #clears the figure

def analyzePeaks(signal, savePath, boxNumber, channel):     #accepts a signal, save path, and box/ch number
    smoothed = smoothWithSavgol(signal, 11, 2)              #smooths the signal to allow more accurate peak detection
    minPeakProm = 0.1                                       #minimum height required to be considered a peak. 0.1=10% of dynamic range (below)    
    smoothPeaks, smoothedDicts = sig.find_peaks(smoothed, prominence=(np.max(smoothed)-np.min(smoothed))*minPeakProm)
    #Identifies peaks based on signal dynamic range and specified peak prominence above. This will struggle to ID peaks in decaying datasets
    
    if len(smoothPeaks) > 0:                                                        #if peaks are detected
        proms, leftBase, rightBase = sig.peak_prominences(smoothed, smoothPeaks)    #returns peak proms               
        widths, heights, leftIndex, rightIndex = sig.peak_widths(smoothed, smoothPeaks, rel_height=0.5) #returns [0]=widths, [1]=heights, [2]=left ips, [3]=right ips (all ndarrays)
        if plotIndividualPeaks == True:                                             #user set at top of script
            print("plotting box analysis from box #" + str(boxNumber) + " for " + channel)  #terminal feedback for script progression
            printBoxPeaks(signal, smoothed, smoothPeaks, heights, leftIndex, rightIndex, proms, savePath, boxNumber, channel)   #calls plot box peaks function
        width = np.mean(widths, axis=0)                     #if multiple peaks are detected, returns the mean
        max = np.mean(smoothed[smoothPeaks], axis=0)        #if multiple peaks are detected, returns the mean
        min = np.mean(smoothed[smoothPeaks]-proms, axis=0)  #if multiple peaks are detected, returns the mean
        amp = max-min
        relAmp = amp/min
        return(width, max, min, amp, relAmp)            #returns peak values
    else:                                               #if no peaks are detected...
        return(np.NaN, np.NaN, np.NaN, np.NaN, np.NaN)  #returns NaNs. 

def plotCF(corArray, savePath, shifts, channel, cfType = "ACF"):        # accepts array of cross or autocorrelation curves (1 for each box), 
                                                                        # a save path, a list of shifts (periods or signal shifts), a channel 
                                                                        # (if applicable), and the type of correlation fxn (cross or auto)
    plotSavePath = savePath / (channel + "mean" + cfType + ".png")      # save path for plot   
    mean = np.nanmean(corArray, axis=0)                                 # mean correlation curve (ignoring nans, i.e. the curves that didn't report a shift)
    std = np.nanstd(corArray, axis=0)                                   # y-axis standard deviation of the curve (i.e., shape of the curve, not stdev of shift values)
    lags = np.arange(-(mean.shape[0]+1)/2+1, (mean.shape[0]+1)/2)       # x axis for correlation plots

    shifts = shifts[2:]                                                 # shift values (just the numbers, not the text)
    if np.isnan(np.max(shifts)) == True:                                # filters out nans if they exit
        shifts = [x for x in shifts if np.isnan(x) != True] 
    
    boxesAndLags = np.vstack((lags, mean, std)).T                       # Makes an ndarray zipping each of the box names (listOfBoxes) and lags for each box (ccfAnswers[1])

    plt.subplot(2,1,1)                                                  # top subplot
    plt.subplots_adjust(wspace=0.4)                                     # adjust horizontal white space
    plt.subplots_adjust(hspace=0.4)                                     # adjust vertical white space
    plt.plot(lags, mean)                                                # plots of the mean correlation function
    plt.fill_between(lags, mean-std, mean+std, alpha = 0.5)             # plots the ±Std Dev as a semi-opaque fill
    plt.xlabel("Average " + cfType + " curve ± Std Dev")                # x-axis label

    plt.subplot(2,2,3)                                                  # bottom left subplot
    plt.hist(shifts)                                                    # histogram of shift values
    if cfType == "ACF":                                                 # for autocorrelations:
        plt.xlabel("Histogram of Period values")                        # x-axis label
    if cfType == "CCF":                                                 # for cross correlation
        plt.xlabel("Histogram of Shift values")                         # x-axis label
    plt.ylabel("Occurrences")                                           # y-axis label
    
    plt.subplot(2,2,4)                                                  # bottom right subplot
    plt.boxplot(shifts)                                                 # boxplot of shift values
    if cfType == "ACF":                                                 # for autocorrelations:
        plt.xlabel("Boxplot of Period values")                          # x-axis label
        plt.ylabel("Measured Period (frames)")                          # y-axis label
    if cfType == "CCF":                                                 # for cross correlation
        plt.xlabel("Boxplot of Shift values")                           # x-axis label
        plt.ylabel("Measured Shift (frames)")                           # y-axis label
    plt.xticks(ticks=[])                                                # empty list for x-axis tick labels (i.e. no labels)

    plt.savefig(plotSavePath, dpi=80)                                   # saves the figure
    plt.close()                                                         # clears the figure
    return(boxesAndLags)                                                # returns an array containing the x axis values, mean autoccorelation values, and the std dev values

def plotPeaks(widthList, minList, maxList, ampList, savePath, channel):             # accepts lists of peak measurements, a save path, and the channel measured
    savePath = savePath / (channel + "MeanPeakMeasurements.png")                    # save path

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))          # figure object w/ three subplots
    fig.subplots_adjust(wspace=0.4)                                                 # adjust subplot horizontal spacing
    ax1.hist(ampList, bins=20, color="tab:purple", label = "amp", alpha = 0.75)     # adds histogram of amplitudes to subplot 1
    ax1.hist(minList, bins=20, color="tab:orange", label = "min", alpha = 0.75)     # adds histogram of min peak values to subplot 1
    ax1.hist(maxList, bins=20, color="tab:blue", label = "max", alpha = 0.75)       # adds histogram of max peak values to subplot 1
    ax1.legend(loc='upper right', fontsize='small', ncol=1)                         # legend
    ax1.set_xlabel("Histogram of peak values")                                      # x axis label
    ax1.set_ylabel("Occurrences")                                                   # y axis label
    
    labels = ["amp", "min", "max"]                                                  # labels to use
    colors = ['tab:purple', 'tab:orange', 'tab:blue']                               # colors to use
    plotThis = [ampList, minList, maxList]                                          # data to plot
    bplot = ax2.boxplot(plotThis, vert=True, patch_artist=True, labels=labels)      # boxplot object
    for patch, color in zip(bplot['boxes'], colors):                                
        patch.set_facecolor(color)                                                  # sets the face color of each box 
    ax2.set_xlabel("Boxplot of peak values")                                        # x axis label
    ax2.set_ylabel("Pixel value (AU)")                                              # y axis label

    ax3.hist(widthList, bins=20, color="tab:blue", label = "max", alpha = 0.75)     # histogram of width values
    ax3.set_xlabel("Histogram of temporal width values")                            # x axis label
    ax3.set_ylabel("Occurrences")                                                   # y axis label

    plt.savefig(savePath, dpi=80)                                                   # saves the figure
    plt.close()                                                                     # closes the figure

def saveBoxValues(measurementList, savePath, columnNames):
    df = pd.DataFrame(measurementList, columns = columnNames)                       # converts the list of lists containing all of the ccf statistics into a pandas dataframe
    fileName = "0_summaryStats.csv"                                                 # file name to save as
    df.to_csv(savePath / fileName, float_format = '%.2f')                           # saves the dataframe to a .csv file

def calcListStats(lis):                                                             # given a list or array
    arr = np.array(lis)                                                             # list to array
    arr = arr[np.logical_not(np.isnan(arr))]                                        # duplicate array, excluding nans
    mean = np.mean(arr)                                                             # mean excluding nans
    median = np.median(arr)                                                         # median excluding nans
    std = np.std(arr)                                                               # standard deviation excluding nans
    sem = std/math.sqrt(arr.shape[0])                                               # standard error of the mean excluding nans
    return(mean, median, std, sem)                                                  # return statistics

def plotComparisons(dataFrame, variable, savePath):
    ax = sns.boxplot(x="Group Name", y=variable, data=dataFrame, palette = "Set2", showfliers = False)		# Makes a boxplot
    ax = sns.swarmplot(x="Group Name", y=variable, data=dataFrame, color=".25")							    # Makes a scatterplot
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)                                                    # x axis labels
    fig = ax.get_figure()																			        # Makes figure object
    fig.savefig((savePath / variable), dpi=300, bbox_inches='tight')						                # saves the plot to the specified file destination	
    plt.close()                                                                                             # close the figure

def makeLog(directory, logParams):                                  # makes a text log with script parameters
    logPath = os.path.join(directory, "log.txt")                    # path to log file
    now = datetime.datetime.now()                                   # get current date and time
    logFile = open(logPath, "w")                                    # initiate text file
    logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")     # write current date and time
    for key, value in logParams.items():                            # for each key:value pair in the parameter dictionary...
        logFile.write('%s: %s\n' % (key, value))                    # write pair to new line
    logFile.close()                                                 # close the file
    
#################################################################
#################################################################
#############                                       #############
#############    FUNCTIONS ABOVE, WORKFLOW BELOW    #############
#############                                       #############
#################################################################
#################################################################

start = timeit.default_timer()
directory = baseDirectory                                       # redundant line...
fileNames = findWorkspace(baseDirectory)             # string object describing the file path, list object containing all file names ending with .tif
masterStatsList = []                                            # empty list to fill with file stats
columnHeaders = []                                              # emtpy list to fill with column headers

''' ** error catching for group names ** '''
groupsFound = []                            # empty list to fill with group names that have been matched
for group in groupNames:                    # check each group
    for file in fileNames:                  # against each file name
        if group in file:                   # and check for a match
            if group not in groupsFound:    # if it's not already in the list of matched groups
                groupsFound.append(group)   # append it

uniqueDic = {}                              # dict of file names and their matches
for file in fileNames:                      # check each file
    uniqueDic[file] = []                    # assign it an empty string
    for group in groupNames:                # check each group
        if group in file:                   # if there's a match...
            uniqueDic[file].append(group)   # ... add it to the list

for val in uniqueDic.values():              # check each file
    if len(val) > 1:                        # each file name should only have ONE matching group name
        print('Error: make sure your group names are unique to each group')
        sys.exit()

if len(groupsFound) != len(groupNames):     # the number of groups should match the number of groups that were matched 
    print('Error: One or more groups not matched. Make sure your group names are present in the file names')
    sys.exit()

makeLog(directory, logParams)                                   # make log text file

for i in range(len(fileNames)):                                 # iterates through the .tif files in the specified directory

    print("Starting to work on " + fileNames[i] + "!")          # user feedback
    imageStack=skio.imread(directory + "/" + fileNames[i])      # reads image as np ndArray
    nameWithoutExtension = fileNames[i].rsplit(".",1)[0]        # gets the file name without the file extension
    boxSavePath = pathlib.Path(directory + "/0_signalProcessing/" + nameWithoutExtension) # sets save path for output for each image file
    boxSavePath.mkdir(exist_ok=True, parents=True)              # makes save path for output, if it doesn't already exist
    if groupNames != ['']:                                      # if user entered group names to compare...
        groupName = setGroups(groupNames, nameWithoutExtension) # return which group this file belongs to

    if imageStack.shape[1] == 2:    # Attempt the verify the number of channels in the image
        imageChannels = 2           # imageStack.shape[1] will either be the number of channels, or the number of pixels on the y-axis
    elif imageStack.ndim == 3:      
        imageChannels = 1           # imageStack.ndim == 3 = the number of dimensions. a 1-channel stack won't have the 4th channel dimension
    else:                           # This should cover most use cases...
        print(nameWithoutExtension + "was NOT processed. Are you sure you have a standard sized image with one or two channels saved in standard `tzcyx` order?")
        continue

#################################################################
#################################################################
#############                                       #############
#############         ONE CHANNEL WORKFLOW          #############
#############                                       #############
#################################################################
#################################################################

    if imageChannels == 1:   
        print("Starting 1-channel workflow")                            # user feedback
        boxMeans = findBoxMeans(imageStack, boxSizeInPx)                # returns array of mean px value in each box; mean box value for every frame in dataset
        numBoxes = boxMeans.shape[0]                                    # returns number of boxes in array (fxn of image dimensions and box size)
        columnNames = ["Parameter", "Mean", "Median", "StdDev", "SEM"]  # initial column names, will be expanded in for loop below
        acfPlots=np.empty((imageStack.shape[0]*2-1))                    # empty array with otherwise the correct shape for an autocorrelation plot, will be appended to below
        
        paramDict = {"Ch1 Period":["Ch1 Period"],                       # dict with a string description of each parameter and empty list to append measurements to
                     "Ch1 Width":["Ch1 Width"],                         # every list has the string description of the measurement in index 0
                     "Ch1 Max":["Ch1 Max"], 
                     "Ch1 Min":["Ch1 Min"],                             #  !!!!! DO I WANT TO KEEP CH1 IN THERE FOR 1-CHANNEL WORKFLOW?
                     "Ch1 Amp":["Ch1 Amp"], 
                     "Ch1 Rel Amp":["Ch1 Rel Amp"]} 
        
        for boxNumber in range(numBoxes):                               # iterates through ndarray of box means
            columnNames.append("Box#" + str(boxNumber))                 # appends the box number to the column names
            acfPlot, period = findACF(boxMeans[boxNumber],              # calculates the acf curve and signal period for every box. 
                                      boxSavePath, 
                                      boxNumber, 
                                      channel = "")                     # channels is empty b/c there's only one channel.
 
            width, max, min, amp, relAmp = analyzePeaks(boxMeans[boxNumber], 
                                                        boxSavePath,    # finds the peak width, max, min, amp, and relAmp for each box
                                                        boxNumber, 
                                                        channel="")     # channels is empty b/c there's only one channel.

            acfPlots = np.vstack((acfPlots, acfPlot))                   # stacks the acf plot for the current box onto the growing array of acf plots
                                                                        # could make this more memory efficient by making the correct sized array first, and redefining for each box
            varDict = {"Ch1 Period":period,                             # dict with string descriptors matching paramDict above
                       "Ch1 Width":width, 
                       "Ch1 Max":max, 
                       "Ch1 Min":min, 
                       "Ch1 Amp":amp, 
                       "Ch1 Rel Amp":relAmp} 
            for key, var in varDict.items():                            # iterates through the dictionary...
                paramDict[key].append(float(var))                       # ...and appends the appropriate variable into the growing lists in paramdict
        
        acfPlots = np.delete(acfPlots, obj=0, axis=0)                   # deletes the empty first obj in the acf plots array (won't need this if I include memory saving step suggested above)
        cfArray = plotCF(acfPlots, boxSavePath, paramDict["Ch1 Period"], channel="")    # plots the mean ± std dev autocorrelation functions and returns an array of the x axis values, mean autoccorelation values, and the std dev values
        df = pd.DataFrame(cfArray, columns=["X Axis", "ACF Mean", "ACF Std Dev"])       # moves the mean±std into a dataframe...
        df.to_csv(boxSavePath / ("cfPlots.csv"))                                        # ... and saves it as a .csv
        
        plotPeaks(paramDict["Ch1 Width"][1:],                           # sends data to the plot peaks fxn to plot peak population histograms etc
                  paramDict["Ch1 Min"][1:],                             # excludes the first index in the list, which is the string description of the measurement
                  paramDict["Ch1 Max"][1:],                         
                  paramDict["Ch1 Amp"][1:],                         
                  boxSavePath,                      
                  channel = "")                                         # channels is empty b/c there's only one channel.
        
        summaryDict = {"Filename":nameWithoutExtension}                 # create dict to fill with summary stats; start with the file name
        summaryDict["# of Boxes"] = numBoxes                            # add number of boxes in the file
        periods = [x for x in paramDict["Ch1 Period"][1:] if np.isnan(x) != True]
                                                                        # removes nans from list of period measurements
                                                                        # find ACF function returns a nan if no suitable peaks are detected
        pcntZeros = ((numBoxes-len(periods))/numBoxes)*100              # calculates what percent of the period measurments were nan (bad measurements)
        summaryDict["Ch1 Pcnt Zero Boxes"] = pcntZeros                  # and appends to dict. This can be used as a metric for analysis quality.
        if groupNames != ['']:                                        
            summaryDict["Group Name"] = groupName               # !!!!! MAY NEED TO MODIFY THIS TO PLAY NICE WITH ANI'S GUI

        for meas in ["Period",                                          # for each major peak measurement...
                     "Width", 
                     "Max", 
                     "Min", 
                     "Amp", 
                     "Rel Amp"]:
            mean, median, std, sem =  calcListStats(paramDict["Ch1 " + meas][1:]) 
                                                                        # pass that corresponding list to the calc list stats fxn
            for index, item in {1:mean,                                 
                                2:median, 
                                3:std,                                  # temp dict that defines which index to insert which measurement into the parameter dict
                                4:sem}.items():
                paramDict["Ch1 " + meas].insert(index, item)            # inserts each measurement at the appropriate location
            for stat, val in {"Mean":mean, 
                              "Median":median,                          # temp dict that defines string description for each variable...
                              "StDev":std, 
                              "SEM":sem}.items():
                summaryDict["Ch1 " + stat + " " + meas] = val           # ...and appends it to the summary dict
        
        for key in summaryDict.keys(): 
            if key not in columnHeaders:                                # if string description of some variable is not already in the column headers list...
                columnHeaders.append(key)                               # ...append it

        listOfMeasurements = []
        for finishedList in paramDict.values():
            listOfMeasurements.append(finishedList)
        saveBoxValues(listOfMeasurements, boxSavePath, columnNames)     # single function that prints summary and box size for everything

        masterStatsList.append(summaryDict)                             # summary of stats for this file is finished, append it to the growing list and move on to the next

        print(str(round((i+1)/len(fileNames)*100, 1)) + 
                  "%" + " Finished with Analysis")                      # user feedback

#################################################################
#################################################################
#############                                       #############
#############         TWO CHANNEL WORKFLOW          #############
#############                                       #############
#################################################################
#################################################################

    if imageChannels == 2:   
        print("Starting 2-channel workflow")                            # user feedback
        subs = np.split(imageStack, 2, 1)                               # list object containing two arrays corresponding to the two channels of the imageStack
        ch1 = np.squeeze(subs[0],axis=1)                                # array object corresponding to channel one of imageStack. Also deletes axis 1, the "channel" axis, which is now empty
        ch2 = np.squeeze(subs[1],axis=1)                                # array object corresponding to channel two of imageStack. Also deletes axis 1, the "channel" axis, which is now empty
        
        ch1BoxMeans = findBoxMeans(ch1, boxSizeInPx)                    # returns array of mean px value in each box; mean box value for every frame in dataset
        ch2BoxMeans = findBoxMeans(ch2, boxSizeInPx)                    # returns array of mean px value in each box; mean box value for every frame in dataset

        numBoxes = ch1BoxMeans.shape[0]                                 # returns number of boxes in array (fxn of image dimensions and box size); same as ch2BoxMeans.shape[0]
        columnNames = ["Parameter", "Mean", "Median", "StdDev", "SEM"]  # initial column names, will be expanded in for loop below
        Ch1AcfPlots = np.zeros((imageStack.shape[0]*2-1))               # empty array with otherwise the correct shape for an autocorrelation plot, will be appended to below
        Ch2AcfPlots = np.zeros((imageStack.shape[0]*2-1))               # empty array with otherwise the correct shape for an autocorrelation plot, will be appended to below
        ccfPlots = np.zeros((imageStack.shape[0]*2-1))                  # empty array with otherwise the correct shape for a crosscorrelation plot, will be appended to below
        paramDict = {"Signal Shift":[],
                     "Ch1 Period":[], 
                     "Ch1 Width":[],                                    # dict with a string description of each parameter and empty list to append measurements to
                     "Ch1 Max":[], 
                     "Ch1 Min":[], 
                     "Ch1 Amp":[], 
                     "Ch1 Rel Amp":[], 
                     "Ch2 Period":[], 
                     "Ch2 Width":[], 
                     "Ch2 Max":[], 
                     "Ch2 Min":[], 
                     "Ch2 Amp":[], 
                     "Ch2 Rel Amp":[]}
        for key, var in paramDict.items():                              # lazy step...
            var.append(key)                                             # every list has the string description of the measurement in index 0
        
        for boxNumber in range(numBoxes):                               # iterates through ndarray of box means
            columnNames.append("Box#" + str(boxNumber))                 # appends the box number to the column names
            ccfPlot, shift = findCCF(ch1BoxMeans[boxNumber],    
                                     ch2BoxMeans[boxNumber],            # calculates the ccf curve and signal shift for every box. 
                                     boxSavePath, 
                                     boxNumber)
            
            acfPlotCh1, periodCh1 = findACF(ch1BoxMeans[boxNumber],     # calculates the acf curve and signal period for every box in channel 1
                                            boxSavePath, 
                                            boxNumber, 
                                            channel = "Ch1")

            widthCh1, maxCh1, minCh1, ampCh1, relAmpCh1 = analyzePeaks(ch1BoxMeans[boxNumber], 
                                                                       boxSavePath,     # finds the peak width, max, min, amp, and relAmp for each box
                                                                       boxNumber, 
                                                                       channel = "Ch1") 
            
            acfPlotCh2, periodCh2 = findACF(ch2BoxMeans[boxNumber],     # calculates the acf curve and signal period for every box in channel 2
                                            boxSavePath, 
                                            boxNumber, 
                                            channel = "Ch2")

            widthCh2, maxCh2, minCh2, ampCh2, relAmpCh2 = analyzePeaks(ch2BoxMeans[boxNumber], 
                                                                       boxSavePath,     # finds the peak width, max, min, amp, and relAmp for each box
                                                                       boxNumber, 
                                                                       channel = "Ch2") 
                                                                       
            ccfPlots = np.vstack((ccfPlots, ccfPlot))                   # stacks the ccf plot for the current box onto the growing array of ccf plots
            Ch1AcfPlots = np.vstack((Ch1AcfPlots, acfPlotCh1))          # stacks the acf plot for the current box onto the growing array of ch1 acf plots
            Ch2AcfPlots = np.vstack((Ch2AcfPlots, acfPlotCh2))          # stacks the acf plot for the current box onto the growing array of ch2 acf plots
            varDict = {"Signal Shift":shift,
                       "Ch1 Period":periodCh1, 
                       "Ch1 Width":widthCh1, 
                       "Ch1 Max":maxCh1, 
                       "Ch1 Min":minCh1,                                # dict with string descriptors matching paramDict above
                       "Ch1 Amp":ampCh1, 
                       "Ch1 Rel Amp":relAmpCh1,
                       "Ch2 Period":periodCh2, 
                       "Ch2 Width":widthCh2, 
                       "Ch2 Max":maxCh2, 
                       "Ch2 Min":minCh2, 
                       "Ch2 Amp":ampCh2, 
                       "Ch2 Rel Amp":relAmpCh2}
            for key, var in varDict.items():                            # iterates through the dictionary...
                paramDict[key].append(float(var))                       # ...and appends the appropriate variable into the growing lists in paramdict

        for grownArray in [Ch1AcfPlots, Ch2AcfPlots, ccfPlots]:         # iterates through the arrays of ccf and acf plots
            grownArray = np.delete(grownArray, obj=0, axis=0)           # deletes the empty array in each of the respective correlation plot arrays
        
        listOfCFs = []                                                  # empty list to fill with raw plot values for the ccf and acf plots

        listOfCFs.append(plotCF(ccfPlots,                               # plots the ccf and returns the raw plot values for the ccf plot
                                boxSavePath, 
                                paramDict["Signal Shift"], 
                                channel = "", 
                                cfType = "CCF"))

        for key, var in {"Ch1 Period":Ch1AcfPlots,                      # temporary dict with string descriptors and variables to work with
                         "Ch2 Period": Ch2AcfPlots}.items():

            listOfCFs.append(plotCF(var,                                # plots the acf for both ch1 and ch2 and returns the raw plot values for the acf plots
                                    boxSavePath, 
                                    paramDict[key], 
                                    channel = key[:3]))
        
        df = pd.DataFrame(np.hstack(listOfCFs), columns=["X Axis",      # converts raw plot values to a dataframe...
                                                         "CCF Mean", 
                                                         "CCF Std Dev", 
                                                         "X Axis", 
                                                         "Ch1 Mean", 
                                                         "Ch1 Std Dev", 
                                                         "X Axis", 
                                                         "Ch2 Mean", 
                                                         "Ch2 Std Dev"])
        df.to_csv(boxSavePath / ("cfPlots.csv"))                        # ... and saves it as a .csv so you can make your own pretty graphs

        periodsCh1 = [x for x in paramDict["Ch1 Period"][1:] if np.isnan(x) != True]
                                                                        # removes nans from list of ch1 period measurements
                                                                        # find ACF function returns a nan if no suitable peaks are detected
        periodsCh2 = [x for x in paramDict["Ch2 Period"][1:] if np.isnan(x) != True]
                                                                        # removes nans from list of ch2 period measurements
                                                                        # find ACF function returns a nan if no suitable peaks are detected
        pcntZerosCh1 = ((numBoxes-len(periodsCh1))/numBoxes)*100        # calculates what percent of the period measurments were nan (bad measurements)
        pcntZerosCh2 = ((numBoxes-len(periodsCh2))/numBoxes)*100        # calculates what percent of the period measurments were nan (bad measurements)

        for ch in ["Ch1", "Ch2"]:                                       # for both the ch1 and ch2 data
            plotPeaks(paramDict[ch + " Width"][1:],                     # sends data to the plot peaks fxn to plot peak population histograms etc
                      paramDict[ch + " Min"][1:],                       # excludes the first index in the list, which is the string description of the measurement
                      paramDict[ch + " Max"][1:], 
                      paramDict[ch + " Amp"][1:], 
                      boxSavePath, channel=ch)

        summaryDict={"Filename":nameWithoutExtension}                   # create dict to fill with summary stats; start with the file name
        summaryDict["# of Boxes"] = numBoxes                            # add number of boxes analyzed in the image
        summaryDict["Ch1 Pcnt Zero Boxes"] = pcntZerosCh1               # This can be used as a metric for analysis quality.
        summaryDict["Ch2 Pcnt Zero Boxes"] = pcntZerosCh2               # This can be used as a metric for analysis quality.

        if groupNames != ['']:                        
            summaryDict["Group Name"] = groupName       # !!!!! MAY NEED TO MODIFY THIS TO PLAY NICE WITH ANI'S GUI

        mean, median, std, sem =  calcListStats(paramDict["Signal Shift"][1:]) 
                                                                        # return the list stats for signal shift
        for stat, val in {"Mean":mean, 
                          "Median":median,                              # temporary dict with string descriptions for each variable
                          "StDev":std,                                  # iterates through each variable...
                          "SEM":sem}.items():
            summaryDict[stat + " Signal Shift"] = val                   # ...and appends it to the summary dict 
        for ch in ["Ch1", "Ch2"]:                                       # for both Ch1 and Ch2...
            for meas in ["Period", 
                         "Width",                                       # and for each type of measurement..
                         "Max", 
                         "Min", 
                         "Amp", 
                         "Rel Amp"]:
                mean, median, std, sem =  calcListStats(paramDict[ch + " " + meas][1:])
                                                                        # define the list stats for that channel and measurement
                for index, item in {1:mean, 
                                    2:median,                           # temp dict that defines which index to insert which measurement into the parameter dict
                                    3:std, 
                                    4:sem}.items():

                    paramDict[ch + " " + meas].insert(index, item)      # inserts each measurement at the appropriate location
                for stat, val in {"Mean":mean, 
                                  "Median":median, 
                                  "StDev":std,                          # temp dict that defines string description for each variable...
                                  "SEM":sem}.items():
                    summaryDict[ch + " " + stat + " " + meas] = val     # ...and appends it to the summary dict
        
        for key in summaryDict.keys():
            if key not in columnHeaders:                                # if string description of some variable is not already in the column headers list...
                columnHeaders.append(key)                               # ...append it

        listOfMeasurements = []                                         
        for finishedList in paramDict.values():
            listOfMeasurements.append(finishedList)
        saveBoxValues(listOfMeasurements, boxSavePath, columnNames)     # single function that prints summary and box size for everything

        masterStatsList.append(summaryDict)                             # summary of stats for this file is finished, append it to the growing list and move on to the next
        print(str(round((i+1)/len(fileNames)*100, 1)) + 
                  "%" + " Finished with Analysis")                      # user feedback
          
#################################################################
#################################################################
#############                                       #############
#############           Saving Analysis             #############
#############                                       #############
#################################################################
#################################################################

df = pd.DataFrame(masterStatsList, columns=columnHeaders)               # convert all the stats to a df
df.to_csv(directory + "/0_fileStats.csv")                               # and export as a .csv


if groupNames != ['']:                                                  # if the user added group names to compare
    compareSavePath = pathlib.Path(directory + "/0_comparisons")        # sets save path for output
    compareSavePath.mkdir(exist_ok=True, parents=True)                  # makes save path for output, if it doesn't already exist
    comparisonsToMake = ["Mean Signal Shift"]                           # manually adding in the mean signal shift, all other comparisons are present in the dicts
    for ch in ["Ch1", "Ch2"]:                                           # for each channel
        for meas in ["Period", 
                     "Width", 
                     "Min",                                             # for each measurement
                     "Max", 
                     "Amp", 
                     "Rel Amp"]:
            comparisonsToMake.append(ch + " Mean " + meas)              # create the measurement
    for comparison in comparisonsToMake:                                # for each comparison
        try:
            plotComparisons(df, comparison, compareSavePath)            # try plotting each comparison
        except ValueError:
            pass                                                        # and skip the ones you don't have

stop = timeit.default_timer()
execution_time = stop - start

print("Program Executed in " + str(round(execution_time, 2)) + " seconds") # It returns time in seconds

