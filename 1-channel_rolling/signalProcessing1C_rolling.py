import os                                    
import sys                                   
import math
import pathlib  
import datetime                             
import numpy as np                           
import pandas as pd 
import tkinter as tk
from tkinter import Tk
from tkinter import ttk
import skimage.io as skio 
import scipy.signal as sig                   
import scipy.fftpack as fft                  
import matplotlib.pyplot as plt  
from tkinter.filedialog import askdirectory

analyzeFrames = 50                                                  #defines the length of submovies #JUPYTER
rollBy = 10                                                          #defines the shift (ie roll) between submovies #SATURN

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
plotSubStackACFs = tk.BooleanVar()          #variable for plotting individual SUBSTACKS
plotIndividualPeaksVar = tk.BooleanVar()    #variable for plotting individual peaks
acfPeakPromVar = tk.DoubleVar()             #variable for peak prominance threshold   
acfPeakPromVar.set(0.1)                     #set default value
jupyter = tk.IntVar()             # analyze frames
jupyter.set(50)   
saturn = tk.IntVar()             # shift frames
saturn.set(5)                      
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

''' !!! '''
''' working here on adding the analyze frames and roll by boxes '''
#boxSize entry widget
boxSizeBox = ttk.Entry(root, width = 3, textvariable=boxSizeVar) #creates box widget
boxSizeBox.grid(column=0, row=1, padx=10, sticky='E') #places widget in frame
boxSizeBox.focus()      #focuses cursor in box
boxSizeBox.icursor(2)   #positions cursor after default input characters
ttk.Label(root, text='Enter grid box size (px)').grid(column=1, row=1, columnspan=2, padx=10, sticky='W') #create label text

#boxSize entry widget
boxSizeBox = ttk.Entry(root, width = 3, textvariable=boxSizeVar) #creates box widget
boxSizeBox.grid(column=0, row=1, padx=10, sticky='E') #places widget in frame
boxSizeBox.focus()      #focuses cursor in box
boxSizeBox.icursor(2)   #positions cursor after default input characters
ttk.Label(root, text='Enter grid box size (px)').grid(column=1, row=1, columnspan=2, padx=10, sticky='W') #create label text

''' !!! '''

#create acfpeakprom entry widget
ttk.Entry(root, width = 3, textvariable=acfPeakPromVar).grid(column=0, row=2, padx=10, sticky='E') #create the widget
ttk.Label(root, text='Enter ACF peak prominence threshold').grid(column=1, row=2, padx=10, sticky='W') #create label text

#create checkbox widgets and labels
ttk.Checkbutton(root, variable=plotIndividualACFsVar).grid(column=0, row=5, sticky='E', padx=15)
ttk.Label(root, text='Plot individual ACFs').grid(column=1, row=5, columnspan=2, padx=10, sticky='W') #plot individual ACFs
ttk.Checkbutton(root, variable=plotSubStackACFs).grid(column=0, row=6, sticky='E', padx=15) #plot individual CCFs
ttk.Label(root, text='Plot substack ACFs').grid(column=1, row=6, columnspan=2, padx=10, sticky='W')

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
plotIndividualCCFs = plotSubStackACFs.get()
plotIndividualPeaks = plotIndividualPeaksVar.get()
acfPeakProm = acfPeakPromVar.get()
directory = folderPath.get() 

#make dictionary of parameters for log file use
logParams = {
    "Box Size(px)" : boxSizeInPx,
    "Base Directory" : directory,
    "ACF Peak Prominence" : acfPeakProm,
    "Plot Individual ACFs" : plotIndividualACFs,
    "Plot substack ACFs" : plotSubStackACFs,
    }

errors = []
if acfPeakProm > 1 :
    errors.append("The ACF peak prominence can not be greater than 1, set 'ACF peak prominence threshold' to a value between 0 and 1. More realistically, a value between 0 and 0.5")
if len(directory) < 1 :
    errors.append("You didn't enter a directory to analyze")

if len(errors) >= 1 :
    print("Error Log:")
    for count, error in enumerate(errors):
        print(count,":", error)
    sys.exit("Please fix errors and try again.") 

'''*** End GUI Window ***'''

def smoothWithSavgol(signal, windowSize, polynomial):               # accepts an array, window size, and polynomial to fit
    return(sig.savgol_filter(signal, windowSize, polynomial))       # returns the smoothed signal

def findBoxMeans(imageArray, boxSize):              #accepts an image array as a parameter, as well as the desired box size
    depth = imageArray.shape[0]                     #number of frames
    yDims = imageArray.shape[1]                     #number of pixels on y-axis
    xDims = imageArray.shape[2]                     #number of pixels on x-axis
    yBoxes = yDims // boxSize                       #returns int result of floor division; number of boxes on the y axis
    xBoxes = xDims // boxSize                       #returns int result of floor division; number of boxes on the x axis
    growingArray = np.zeros((1, depth))             #makes a starting array of 64 bit zeros that can be added onto later. shape = (1, depth of imageStack)

    for x in range(xBoxes):                         #iterates through the number of boxes on the x-axis
        for y in range (yBoxes):                    #iterates through the number of boxes on the y-axis
            boxMean = np.array([np.mean(imageArray[:,(y*boxSize):(y*boxSize+boxSize),(x*boxSize):(x*boxSize+boxSize)], (1,2))])  
            #creates a 2d array of shape (depth, 1) containing the mean values of the px within the box for each slices
            growingArray = np.append(growingArray, boxMean, axis = 0)      #appends the 2d array onto the growing array
    growingArray = np.delete(growingArray, 0, axis=0)
    return(growingArray)                            #returns ndarray of shape (number of boxes, number of frames)

def printBoxACF(signal, acor, boxNum, directory, boxNumber, delay=None, subStackIndex=None):
    subStackName = "Frames_" + subStackIndex
    subFolder = "ACF_Plots"
    acfSavePath = directory / subStackName / subFolder
    acfSavePath.mkdir(exist_ok=True, parents=True) 

    xAxis = np.arange(signal.shape[0])                 #np array containing ascending integers up to the value of npts
    lags = np.arange(-signal.shape[0] + 1, signal.shape[0])       #np array ranging from - to + npts

    fig, axs = plt.subplots(nrows=2)
    fig.subplots_adjust(hspace=0.4)
    ax = axs[0]
    ax.plot(xAxis, signal)
    ax.set_ylabel('Mean box px value')
    ax.set_xlabel('Time (frames)')

    if delay == None:
        ax = axs[1]
        ax.plot(lags, acor)
        ax.set_ylabel('auto-correlation')
        ax.set_xlabel("Periodic signal not detected")
        graphName = "boxNo" + str(boxNum) + ".png"
        boxName = acfSavePath / graphName
        plt.savefig(boxName, dpi=75, )            #saves the figure
        plt.clf()
        plt.close(fig)
    else:
        ax = axs[1]
        ax.plot(lags, acor)
        ax.set_ylabel('auto-correlation')
        ax.set_xlabel("Period is " + str(delay) + " frames")
        plt.axvline(x=delay, alpha = 0.5, c = 'red', linestyle = '--')
        plt.axvline(x=-delay, alpha = 0.5, c = 'red', linestyle = '--')
        graphName = "boxNo" + str(boxNum) + ".png"
        boxName = acfSavePath / graphName
        plt.savefig(boxName, dpi=75, )                            #saves the figure
        plt.clf()
        plt.close(fig)

def findACF(signal, directory, imageName, boxNumber, subStackIndex=None):  #accepts a single array (one) channels), which will be correlated to itself.
    npts = signal.shape[0]                      #number of points is the depth of the 0 axis, which is the number of frames in the image
    acov = np.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')    #compute full autocorrelation
    acor = acov / (npts * signal.std() ** 2)                                #normalizes the crosscorr from -1 to +1      
    peaks, dict = sig.find_peaks(acor)#, prominence=0.075)             #ndarray with location of local maxima using scipy.signal.find_peaks
    peaksDiff = abs(peaks - acor.shape[0]//2)                   #ndarray with the absolute difference between each peak and the middle value of the ccor array

    try:
        delay = np.min(peaksDiff[np.nonzero(peaksDiff)])        #numpy.int64 reporting difference between the first peak and zero
        if plotIndividualACFs == True:
            print("plotting box #" + str(boxNumber))
            printBoxACF(signal, acor, boxNumber, directory, boxNumber, delay, subStackIndex)
    except ValueError:
        if plotIndividualACFs == True:
            print("plotting box #" + str(boxNumber))
            printBoxACF(signal, acor, boxNumber, directory, boxNumber, delay=None, subStackIndex=subStackIndex)
        zeroArray = np.full((npts*2-1), np.nan)        #array of nan values w/ shape (x,) where x = the number of points in the ccor
        return(zeroArray, np.nan)
    
    return(acor, delay)

def printPeaks(raw, smoothed, smoothPeaks, heights, leftIndex, rightIndex, proms, directory, boxNumber, subStackIndex):
    subStackName = "Frames_" + subStackIndex
    subFolder = "Peak_Plots"
    peaksSavePath = directory / subStackName / subFolder
    peaksSavePath.mkdir(exist_ok=True, parents=True) 

    x = np.arange(raw.shape[0])
    fig, axs = plt.subplots(nrows=2)
    ax = axs[0]
    ax.plot(x,raw, color='darkcyan', label='raw')
    ax.plot(x,smoothed, color='chocolate', label='smoothed')
    ax=axs[1]
    ax.plot(x,smoothed, color='chocolate', label='smoothed')
    for i in range(smoothPeaks.shape[0]):
        ax.hlines(heights[i], leftIndex[i], rightIndex[i], color='black', alpha = 0.75, linestyle = '-')
        ax.vlines(smoothPeaks[i], smoothed[smoothPeaks[i]]-proms[i], smoothed[smoothPeaks[i]], color='black', alpha = 0.75, linestyle = '-')
    ax.legend(loc='upper right', fontsize='small', ncol=1)  
    graphName = "boxNo" + str(boxNumber) + ".png"
    boxName = peaksSavePath / graphName
    plt.savefig(boxName, dpi=75, )                                        #saves the figure
    plt.clf()
    plt.close(fig)

def analyzePeaks(signal, subStackSavePath, boxNumber, subStackIndex = None):
    smoothed = smoothWithSavgol(signal, 11, 2)
    minVal = np.min(signal)
    maxVal = np.max(signal)
    smoothPeaks, smoothedDicts = sig.find_peaks(smoothed, prominence=(maxVal-minVal)*0.1)
    if len(smoothPeaks) > 0:
        proms, leftBase, rightBase = sig.peak_prominences(smoothed, smoothPeaks)                  
        widths, heights, leftIndex, rightIndex = sig.peak_widths(smoothed, smoothPeaks, rel_height=0.5) #returns [0]=widths, [1]=heights, [2]=left ips, [3]=right ips (all ndarrays)
        if plotIndividualPeaks == True:
            printPeaks(signal, smoothed, smoothPeaks, heights, leftIndex, rightIndex, proms, subStackSavePath, boxNumber, subStackIndex)
        width = np.mean(widths, axis=0)
        max = np.mean(smoothed[smoothPeaks], axis=0)
        min = np.mean(smoothed[smoothPeaks]-proms, axis=0)
        amp = max-min
        relAmp = amp/min
        return(width, max, min, amp, relAmp)
    else:
        return(np.NaN, np.NaN, np.NaN, np.NaN, np.NaN)  #returns NaNs. 

def subStackPlotsAndShifts(acorArray, subStackSavePath, subStackIndex, periods):              #accepts a tuple containing acfs and shifts, which will be plotted and saved
    plotSavePath = subStackSavePath / ("Frames_" + str(subStackIndex) + "_meanAcf.png")
    txtPath = subStackSavePath / ("Frames_" + str(subStackIndex) + "_meanAcf.txt")
    meanAcf = np.nanmean(acorArray, axis=0)
    stdAcf = np.nanstd(acorArray, axis=0)
    lags = np.arange(-(meanAcf.shape[0]+1)/2+1, (meanAcf.shape[0]+1)/2)

    periods = periods[2:]
    if np.isnan(np.max(periods)) == True:                       #filters out nans if they exit
        periods = [x for x in periods if np.isnan(x) != True] 
    
    boxesAndLags = np.vstack((lags, meanAcf, stdAcf))      #Makes an ndarray zipping each of the box names (listOfBoxes) and lags for each box (ccfAnswers[1])
    np.savetxt(txtPath, boxesAndLags, fmt='%1.0f')                                  #saves the ndarray as a text file

    plt.subplot(2,1,1)                                                      #top subplot
    plt.subplots_adjust(wspace=0.4)                                         #adjust horizontal white space
    plt.subplots_adjust(hspace=0.4)                                         #adjust vertical white space
    plt.plot(lags, meanAcf)                                             #plots of the mean CCF
    plt.fill_between(lags, meanAcf-stdAcf, meanAcf+stdAcf, alpha = 0.5) #plots the ±Std Dev
    plt.xlabel("Average ACF curve ± Std Dev")             #x-axis label

    plt.subplot(2,2,3)                                                      #bottom left subplot
    plt.hist(periods)                                                  #histogram of shift values
    plt.xlabel("Histogram of shift values")                                 #x-axis label
    plt.ylabel("Occurrences")                                               #y-axis label
    
    print(subStackSavePath)
    print(subStackIndex)
    print(periods)
    plt.subplot(2,2,4)                                                      #bottom right subplot
    plt.boxplot(periods)                                               #boxplot of shift values
    plt.xlabel("Boxplot of shift values")                                   #x-axis label
    plt.ylabel("Measured shift")                                            #y-axis label
    plt.xticks(ticks=[])                                                    #empty list for x-axis tick labels (i.e. no labels)

    plt.savefig(plotSavePath)                                                  #saves the figure
    plt.close()                                                               #clears the figure

def calcListStats(list, numberOfBoxes):
    npList = np.array(list)
    mean = np.nanmean(npList)
    median = np.nanmedian(npList)
    std = np.nanstd(npList)
    sem = std/math.sqrt(npList.shape[0])
    pcntZeros = ((numBoxes-len(list))/numBoxes)*100
    return(mean, median, std, sem, pcntZeros)

def saveSubStackValues(measurementList, subStackSavePath, valueName, columnNames):
    df = pd.DataFrame(measurementList, columns = columnNames)                                   #converts the list of lists containing all of the ccf statistics into a pandas dataframe
    fileName = valueName + "_Raw.csv"
    df.to_csv(subStackSavePath / fileName)                              #saves the dataframe to a .csv file

def printTemporalVars(listOfVars, variable, subStackSavePath):
    graphName = subStackSavePath / variable
    xAxis=[]
    mean=[]
    std=[]
    bottomBoundary=[]
    topBoundary=[]
    for eachList in listOfVars:
        subStackName = eachList[1].split("_")[0]
        xAxis.append(subStackName)
        mean.append(np.nanmean(eachList[2:]))
        std.append(np.nanstd(eachList[2:]))

    for val1, val2, in zip(mean, std):
        bottomBoundary.append(val1-val2)
        topBoundary.append(val1+val2)
    plt.plot(xAxis, mean)
    plt.fill_between(xAxis, bottomBoundary, topBoundary, alpha=0.5)
    plt.xlabel("Time (frames)")
    plt.ylabel(variable)
    plt.savefig(graphName)                                                  #saves the figure
    plt.close() 

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

fileNames = [fname for fname in os.listdir(directory) if fname.endswith('.tif')]    # list object containing all file names ending with .tif
makeLog(directory, logParams)                                                       # make log text file

for i in range(len(fileNames)):                                 #iterates through the .tif files in the specified directory
    print("Starting to work on " + fileNames[i] + "!")
    nameWithoutExtension = fileNames[i].split(".")[0]
    imageStack=skio.imread(directory + "/" + fileNames[i])      #reads image as ndArray
    assert imageStack.ndim == 3, "Make sure that you dataset has no more than one channel, no more than one z plane, and no less than two time points"

    print("Starting rolling analysis")                                                                 #empty list that will later be populated with shift ACF statistics
    subStackSavePath = pathlib.Path(directory + "/" + nameWithoutExtension + "_subStackAnalysis")                                #path object for a subdirectory to save the ccf data to
    subStackSavePath.mkdir(exist_ok=True, parents=True)                                                   #makes path, ignores error if path already exists
    numberSubMovies = (imageStack.shape[0]-analyzeFrames)//rollBy+1     #calculates the number of times a sub-movie can be evenly created
    paramDict = {"period":[], "width":[], "max":[], "min":[], "amp":[], "relAmp":[]}
    
    for y in range(numberSubMovies):                                    #iterates through each of the possible submovies

        tempParamDict = {"period":[], "width":[], "max":[], "min":[], "amp":[], "relAmp":[]}

        startingFrame = (y*rollBy)                                      #starting frame to duplicate
        endingFrame = (analyzeFrames+(rollBy*y))                        #ending frame to duplicate
        subStack = np.copy(imageStack[startingFrame:endingFrame, ])     #copies the ndarray from starting to ending frame
        subStackIndex = str(startingFrame) + "_" + str(endingFrame)      #string object carrying the sub movie name        
        subStackMeans = findBoxMeans(subStack, boxSizeInPx)
        numBoxes = subStackMeans.shape[0]
        columnNames = ["Parameter", "subStack Index"]
        tempMeanAcf=np.empty((analyzeFrames*2-1))
        
        for key, tempList in tempParamDict.items():
            tempList.append(key) #BEFORE ANALYZING ANY BOXES, EACH PARAM LIST STARTS WITH THE SUBSTACK INDEX
            tempList.append(subStackIndex)
        
        for boxNumber in range(numBoxes):                         #iterates through ndarray of box means
            columnNames.append("Box#" + str(boxNumber))
            acfPlot, period = findACF(subStackMeans[boxNumber], subStackSavePath, nameWithoutExtension, boxNumber, subStackIndex)
            width, max, min, amp, relAmp = analyzePeaks(subStackMeans[boxNumber], subStackSavePath, boxNumber, subStackIndex)
            tempMeanAcf=np.vstack((tempMeanAcf, acfPlot)) #ADDS ONTO THE GROWING LIST OF BOX ACFS FOR EACHTIME POINT

            varDict = {"period":period, "width":width, "max":max, "min":min, "amp":amp, "relAmp":relAmp}
            for key, var in varDict.items():
                tempParamDict[key].append(float(var))
            
            if plotIndividualPeaks == True:
                print(str(round((boxNumber+1)/numBoxes*100, 1)) + "%" + " Finished with frames" + str(subStackIndex))
            elif plotIndividualACFs == True:
                print(str(round((boxNumber+1)/numBoxes*100, 1)) + "%" + " Finished with frames" + str(subStackIndex))
    
        for key, var in tempParamDict.items():
            paramDict[key].append(tempParamDict[key])
        
        meanAcfArray = np.delete(tempMeanAcf, obj=0, axis=0)
        if plotSubStackACFs == True:
            subStackPlotsAndShifts(meanAcfArray, subStackSavePath, subStackIndex, tempParamDict["period"])

        print(str(round((y+1)/numberSubMovies*100, 1)) + "%" + " Finished with " + fileNames[i])
    
    for key, lol in paramDict.items():
        for lis in lol:
            mean, median, std, sem, pcntZeros = calcListStats(lis[2:], numBoxes)
            for index, item in {2:mean, 3:median, 4:std, 5:sem, 6:pcntZeros}.items():
                lis.insert(index, item)

    for index, item in {2:"mean", 3:"median", 4:"std", 5:"sem", 6:"pcntZeros"}.items():
        columnNames.insert(index, item)

    for key, permList in paramDict.items():
        saveSubStackValues(permList, subStackSavePath, key, columnNames)
        printTemporalVars(permList, key, subStackSavePath)

    print(str(round((i+1)/len(fileNames)*100, 1)) + "%" + " Finished with Analysis")