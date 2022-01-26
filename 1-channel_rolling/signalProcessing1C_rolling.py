import os                                    
import sys                                   
import math
import pathlib                               
import numpy as np                           
import pandas as pd 
import skimage.io as skio 
import scipy.signal as sig                   
import scipy.fftpack as fft                  
import matplotlib.pyplot as plt  
from tkinter.filedialog import askdirectory

boxSizeInPx = 20                #ENTER DESIRED BOXED SIZE HERE
plotIndividualACFs = False      #TRUE = PLOTS BOXES; FALSE = ONLY PLOTS POP MEANS
plotIndividualPeaks = False
plotSubStackACFs = False
smoothACF = True                #smooths the ACF to better eliminate noisy peaks
smoothMySignal = True        #TRUE = SMOOTHS THE BOX MEANS PRIOR TO CALCULATING THE WAVE AMPLITUDE AND WIDTHS
smoothingMethod = "savgol"  #"savgol" calls the smoothWithSavgol (polynomial fit); "fft" calls smoothWithFFT (low pass filter)
analyzeFrames = 50                                                  #defines the length of submovies
rollBy = 10                                                          #defines the shift (ie roll) between submovies
baseDirectory = "/Users/bementmbp/Desktop/testing"         #BASE DIRECTORY FOR THE GUI

def findWorkspace(directory, prompt):                                                       #accepts a starting directory and a prompt for the GUI
    targetWorkspace = askdirectory(initialdir=directory, message=prompt)                    #opens prompt asking for folder, keep commented to default to baseDirectory
    #targetWorkspace = directory                                                            #comment this out later if you want a GUI
    filelist = [fname for fname in os.listdir(targetWorkspace) if fname.endswith('.tif')]   #Makes a list of file names that end with .tif
    return(targetWorkspace, filelist)                                                       #returns the folder path and list of file names

def smoothWithFFT(signal, factor):  # the scaling of "span" is open to suggestions
    w = fft.rfft(signal)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-factor / 100000)))
    w[cutoff_idx] = 0
    smoothedSignal = fft.irfft(w)
    return smoothedSignal

def smoothWithSavgol(signal, windowSize, polynomial):  
    smoothedSignal = sig.savgol_filter(signal, windowSize, polynomial)
    return smoothedSignal

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
        plt.savefig(boxName, dpi=75, )                                                  #saves the figure                                                 #saves the figure
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
        plt.savefig(boxName, dpi=75, )                                                  #saves the figure                                                 #saves the figure
        plt.clf()
        plt.close(fig)

def findACF(signal, directory, imageName, boxNumber, subStackIndex=None):  #accepts a single array (one) channels), which will be correlated to itself.
    npts = signal.shape[0]                      #number of points is the depth of the 0 axis, which is the number of frames in the image
    acov = np.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')    #compute full autocorrelation
    acor = acov / (npts * signal.std() ** 2)                                #normalizes the crosscorr from -1 to +1      
    if smoothACF == True:
        acor = smoothWithSavgol(acor, windowSize=3, polynomial=1)
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
    plt.savefig(boxName, dpi=75, )                                                  #saves the figure                                                 #saves the figure
    plt.clf()
    plt.close(fig)

def analyzePeaks(signal, subStackSavePath, boxNumber, subStackIndex = None):
    if smoothMySignal == True:
        if smoothingMethod == "savgol":
            smoothed = smoothWithSavgol(signal, 11, 2)
        elif smoothingMethod == "fft":
            smoothed = smoothWithFFT(signal, 0.4)
        else:
            sys.exit("Choose either 'savgol' or 'fft' as a signal smoothing method")
        minVal = np.min(signal)
        maxVal = np.max(signal)
        xAxis = np.arange(len(signal))
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
        
    else:
        sys.exit("Keep 'smoothMySignal =True' for now")

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

############# FUNCTIONS ABOVE, WORKFLOW BELOW #############

directory, fileNames = findWorkspace(baseDirectory, "PLEASE SELECT YOUR SOURCE WORKSPACE")  #string object describing the file path, list object containing all file names ending with .tif

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