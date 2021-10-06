import os                                       
import sys                                      
import math
import pathlib                                  
import numpy as np
import pandas as pd  
import seaborn as sns
import skimage.io as skio  
import scipy.signal as sig                        
import matplotlib.pyplot as plt    
from tkinter.filedialog import askdirectory              

boxSizeInPx = 20                #Desired box size for analysis
plotIndividualACFs = False      #True = plots signal trace and ACF curve for every box; False = only plots pop means. 
plotIndividualCCFs = False      #True = plots signal trace and CCF curve for every box; False = only plots pop means. 
plotIndividualPeaks = False     #True = plots signal trace and peak picking for every box; False = only plots pop statistics.
compareFiles = True            #True = generates plots comparing the different groups in your dataset; False = only writes wave stats
fileNameIndex = -1              #Necessary for "compareFiles = True", identifies the group index in the filename.
acfPeakProm = 0.1               #Minimum peak prominence to choose in an ACF, set 0-1. Larger values are more stringent. 
baseDirectory = "/Users/bementmbp/Desktop/ccfTest1"      #Base directory for the GUI. Can hard code file path by commenting line 23 and uncommenting line 24. 

def findWorkspace(directory, prompt):                                                       #accepts a starting directory and a prompt for the GUI
    #targetWorkspace = askdirectory(initialdir=directory, message=prompt)                    #opens prompt asking for folder, keep commented to default to baseDirectory
    targetWorkspace = directory                                                            #comment this out later if you want a GUI
    filelist = [fname for fname in os.listdir(targetWorkspace) if fname.endswith('.tif')]   #Makes a list of file names that end with .tif
    return(targetWorkspace, filelist)                                                       #returns the folder path and list of file names

def smoothWithSavgol(signal, windowSize, polynomial):                       #accepts a signal array (or list), number of values to match, and polynomial number.
    smoothedSignal = sig.savgol_filter(signal, windowSize, polynomial)      #smooths the input signal...
    return smoothedSignal                                                   #...and returns it

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
    growingArray = np.delete(growingArray, 0, axis=0)           #deletes the zero array used to initialize
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
    #subFolder =                                                  #Specifies subfolder name
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

def plotCF(corArray, savePath, shifts, channel, cfType = "ACF"):              #accepts a tuple containing acfs and shifts, which will be plotted and saved
    plotSavePath = savePath / (channel + "mean" + cfType + ".png")
    mean = np.nanmean(corArray, axis=0)
    std = np.nanstd(corArray, axis=0)
    lags = np.arange(-(mean.shape[0]+1)/2+1, (mean.shape[0]+1)/2)

    shifts = shifts[2:]
    if np.isnan(np.max(shifts)) == True:                       #filters out nans if they exit
        shifts = [x for x in shifts if np.isnan(x) != True] 
    
    boxesAndLags = np.vstack((lags, mean, std)).T      #Makes an ndarray zipping each of the box names (listOfBoxes) and lags for each box (ccfAnswers[1])
    #np.savetxt(txtPath, boxesAndLags, fmt='%10.5f', delimiter=',')                                  #saves the ndarray as a text file

    plt.subplot(2,1,1)                                                      #top subplot
    plt.subplots_adjust(wspace=0.4)                                         #adjust horizontal white space
    plt.subplots_adjust(hspace=0.4)                                         #adjust vertical white space
    plt.plot(lags, mean)                                             #plots of the mean CCF
    plt.fill_between(lags, mean-std, mean+std, alpha = 0.5) #plots the ±Std Dev
    plt.xlabel("Average " + cfType + " curve ± Std Dev")             #x-axis label

    plt.subplot(2,2,3)                                                      #bottom left subplot
    plt.hist(shifts)                                                  #histogram of shift values
    if cfType == "ACF":
        plt.xlabel("Histogram of Period values")                                 #x-axis label
    if cfType == "CCF":
        plt.xlabel("Histogram of Shift values")
    plt.ylabel("Occurrences")                                               #y-axis label
    
    plt.subplot(2,2,4)                                                      #bottom right subplot
    plt.boxplot(shifts)                                               #boxplot of shift values
    if cfType == "ACF":
        plt.xlabel("Boxplot of Period values")                                 #x-axis label
        plt.ylabel("Measured Period (frames)")
    if cfType == "CCF":
        plt.xlabel("Boxplot of Shift values")
        plt.ylabel("Measured Shift (frames)")                                            #y-axis label
    plt.xticks(ticks=[])                                                    #empty list for x-axis tick labels (i.e. no labels)

    plt.savefig(plotSavePath, dpi=80)                                                  #saves the figure
    plt.close()                                                               #clears the figure
    return(boxesAndLags)

def plotPeaks(widthList, minList, maxList, ampList, savePath, channel):
    savePath = savePath / (channel + "MeanPeakMeasurements.png")

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.4)
    fig.subplots_adjust(wspace=0.4)
    ax1.hist(ampList, bins=20, color="tab:purple", label = "amp", alpha = 0.75)
    ax1.hist(minList, bins=20, color="tab:orange", label = "min", alpha = 0.75)
    ax1.hist(maxList, bins=20, color="tab:blue", label = "max", alpha = 0.75)
    ax1.legend(loc='upper right', fontsize='small', ncol=1) 
    ax1.set_xlabel("Histogram of peak values")
    ax1.set_ylabel("Occurrences")
    
    labels = ["amp", "min", "max"]
    colors = ['tab:purple', 'tab:orange', 'tab:blue']
    plotThis = [ampList, minList, maxList]
    bplot = ax2.boxplot(plotThis, vert=True, patch_artist=True, labels=labels)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_xlabel("Boxplot of peak values")
    ax2.set_ylabel("Pixel value (AU)")

    ax3.hist(widthList, bins=20, color="tab:blue", label = "max", alpha = 0.75)
    ax3.set_xlabel("Histogram of temporal width values")
    ax3.set_ylabel("Occurrences")

    plt.savefig(savePath, dpi=80)                                                  #saves the figure
    plt.close()

def saveBoxValues(measurementList, savePath, columnNames):
    df = pd.DataFrame(measurementList, columns = columnNames)                                   #converts the list of lists containing all of the ccf statistics into a pandas dataframe
    fileName = "0_summaryStats.csv"
    df.to_csv(savePath / fileName, float_format = '%.2f')                              #saves the dataframe to a .csv file

def calcListStats(list):
    npList = np.array(list)
    mean = np.nanmean(npList)
    median = np.nanmedian(npList)
    std = np.nanstd(npList)
    sem = std/math.sqrt(npList.shape[0])
    return(mean, median, std, sem)

def plotComparisons(dataFrame, variable, savePath):
    ax = sns.boxplot(x="Group Name", y=variable, data=dataFrame, palette = "Set2", showfliers = False)		#Makes a boxplot
    ax = sns.swarmplot(x="Group Name", y=variable, data=dataFrame, color=".25")							#Makes a scatterplot
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    fig = ax.get_figure()																			#Makes figure object
    fig.savefig((savePath / variable), dpi=300, bbox_inches='tight')						#saves the plot to the specified file destination	
    plt.close()

############# FUNCTIONS ABOVE, WORKFLOW BELOW #############











directory, fileNames = findWorkspace(baseDirectory, "PLEASE SELECT YOUR SOURCE WORKSPACE")  #string object describing the file path, list object containing all file names ending with .tif
masterStatsList = []
columnHeaders = []

for i in range(len(fileNames)):                                 #iterates through the .tif files in the specified directory

    print("Starting to work on " + fileNames[i] + "!")
    imageStack=skio.imread(directory + "/" + fileNames[i])      #reads image as ndArray
    nameWithoutExtension = fileNames[i].rsplit(".",1)[0]
    boxSavePath = pathlib.Path(directory + "/0_signalProcessing/" + nameWithoutExtension) #sets save path for output
    boxSavePath.mkdir(exist_ok=True, parents=True)  #makes save path for output, if it doesn't already exist
    if compareFiles == True:
        groupName = nameWithoutExtension.split("_")[fileNameIndex]
    
    """Attempt the verify the number of channels in the image"""
    if imageStack.shape[1] == 2:    #imageStack.shape[1] will either be the number of channels, or the number of pixels on the y-axis
        imageChannels = 2          
    elif imageStack.ndim == 3:  #imageStack.ndim == 3 = the number of dimensions. a 1-channel stack won't have the 4th channel dimension
        imageChannels = 1
    else:
        sys.exit("Are you sure you have a standard sized image with one or two channels?")








    
    '''                     ************************ ONE CHANNEL WORKFLOW ************************                  '''
    if imageChannels == 1:   
        print("Starting 1-channel workflow")
        boxMeans = findBoxMeans(imageStack, boxSizeInPx) #returns array of mean px value in each box
        numBoxes = boxMeans.shape[0]                        #returns number of boxes in array
        columnNames = ["Parameter", "Mean", "Median", "StdDev", "SEM"]                         #column names, will be expanded in for loop below
        acfPlots=np.empty((imageStack.shape[0]*2-1))        #
        
        paramDict = {"Ch1 Period":[], "Ch1 Width":[], "Ch1 Max":[], "Ch1 Min":[], "Ch1 Amp":[], "Ch1 Rel Amp":[]} #dictionary object with a string description and empty list to append measurements to
        for key, tempList in paramDict.items():
            tempList.append(key)                    #every list now has the string description of the measurement in index 0
        
        for boxNumber in range(numBoxes):                         #iterates through ndarray of box means
            columnNames.append("Box#" + str(boxNumber))
            acfPlot, period = findACF(boxMeans[boxNumber], boxSavePath, boxNumber, channel = "")
            width, max, min, amp, relAmp = analyzePeaks(boxMeans[boxNumber], boxSavePath, boxNumber, channel="") #calls analyze peaks function, returns width, max, min, amp, relAmp as numpy.float64 objects
            acfPlots = np.vstack((acfPlots, acfPlot))                                                   #ADDS ONTO THE GROWING LIST OF BOX ACFS
            varDict = {"Ch1 Period":period, "Ch1 Width":width, "Ch1 Max":max, "Ch1 Min":min, "Ch1 Amp":amp, "Ch1 Rel Amp":relAmp} #dictionary with string descriptors matching paramDict above
            for key, var in varDict.items():        #iterates through the dictionary...
                paramDict[key].append(float(var))   #...and appends the appropriate variable into the growing lists in paramdict
        
        acfPlots = np.delete(acfPlots, obj=0, axis=0)
        cfArray = plotCF(acfPlots, boxSavePath, paramDict["Ch1 Period"], channel="")
        df = pd.DataFrame(cfArray, columns=["X Axis", "ACF Mean", "ACF Std Dev"])
        df.to_csv(boxSavePath / ("cfPlots.csv"))
        
        plotPeaks(paramDict["Ch1 Width"][1:], paramDict["Ch1 Min"][1:], paramDict["Ch1 Max"][1:], paramDict["Ch1 Amp"][1:], boxSavePath, channel = "")
        
        summaryDict={"Filename":nameWithoutExtension}
        summaryDict["# of Boxes"] = numBoxes
        periods = [x for x in paramDict["Ch1 Period"][1:] if np.isnan(x) != True]
        pcntZeros = ((numBoxes-len(periods))/numBoxes)*100
        summaryDict["Ch1 Pcnt Zero Boxes"] = pcntZeros
        if compareFiles == True:
            summaryDict["Group Name"] = groupName

        for meas in ["Period", "Width", "Max", "Min", "Amp", "Rel Amp"]:
            mean, median, std, sem =  calcListStats(paramDict["Ch1 " + meas][1:])
            for index, item in {1:mean, 2:median, 3:std, 4:sem}.items():
                paramDict["Ch1 " + meas].insert(index, item)
            for stat, val in {"Mean":mean, "Median":median, "StDev":std, "SEM":sem}.items():
                summaryDict["Ch1 " + stat + " " + meas] = val
        
        for key in summaryDict.keys():
            if key not in columnHeaders:
                columnHeaders.append(key)

        listOfMeasurements = []
        for finishedList in paramDict.values():
            listOfMeasurements.append(finishedList)
        saveBoxValues(listOfMeasurements, boxSavePath, columnNames)  #SINGLE FUNCTION THAT PRINTS SUMMARY AND BOX SIZE FOR EVERYTHING

        masterStatsList.append(summaryDict)

        print(str(round((i+1)/len(fileNames)*100, 1)) + "%" + " Finished with Analysis")









    '''                     ************************  TWO CHANNEL WORKFLOW ************************                  '''
    if imageChannels == 2:   
        print("Starting 2-channel workflow")
        subs = np.split(imageStack, 2, 1) #List object containing two arrays corresponding to the two channels of the imageStack
        ch1 = np.squeeze(subs[0],axis=1)  #array object corresponding to channel one of imageStack. Also deletes axis 1, the "channel" axis, which is now empty
        ch2 = np.squeeze(subs[1],axis=1)  #array object corresponding to channel two of imageStack. Also deletes axis 1, the "channel" axis, which is now empty
        
        ch1BoxMeans = findBoxMeans(ch1, boxSizeInPx) #ndarray of shape (# boxes, # frames)
        ch2BoxMeans = findBoxMeans(ch2, boxSizeInPx) #ndarray of shape (# boxes, # frames)

        assert ch2BoxMeans.size == ch1BoxMeans.size, "ch1BoxMeans and ch2BoxMeans are not the same size, something went horribly wrong"
        numBoxes = ch1BoxMeans.shape[0]
        Ch1AcfPlots = np.zeros((imageStack.shape[0]*2-1))
        Ch2AcfPlots = np.zeros((imageStack.shape[0]*2-1))
        ccfPlots = np.zeros((imageStack.shape[0]*2-1))
        paramDict = {"Signal Shift":[],
                     "Ch1 Period":[], "Ch1 Width":[], "Ch1 Max":[], "Ch1 Min":[], "Ch1 Amp":[], "Ch1 Rel Amp":[], 
                     "Ch2 Period":[], "Ch2 Width":[], "Ch2 Max":[], "Ch2 Min":[], "Ch2 Amp":[], "Ch2 Rel Amp":[]}
        for key, var in paramDict.items():
            var.append(key) 
        columnNames = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
        for boxNumber in range(numBoxes):                         #iterates through ndarray of box means
            columnNames.append("Box#" + str(boxNumber))
            ccfPlot, shift = findCCF(ch1BoxMeans[boxNumber], ch2BoxMeans[boxNumber], boxSavePath, boxNumber)
            acfPlotCh1, periodCh1 = findACF(ch1BoxMeans[boxNumber], boxSavePath, boxNumber, channel = "Ch1")
            widthCh1, maxCh1, minCh1, ampCh1, relAmpCh1 = analyzePeaks(ch1BoxMeans[boxNumber], boxSavePath, boxNumber, channel = "Ch1") #calls analyze peaks function, returns width, max, min, amp, relAmp as numpy.float64 objects
            acfPlotCh2, periodCh2 = findACF(ch2BoxMeans[boxNumber], boxSavePath, boxNumber, channel = "Ch2")
            widthCh2, maxCh2, minCh2, ampCh2, relAmpCh2 = analyzePeaks(ch2BoxMeans[boxNumber], boxSavePath, boxNumber, channel = "Ch2") #calls analyze peaks function, returns width, max, min, amp, relAmp as numpy.float64 objects
            ccfPlots = np.vstack((ccfPlots, ccfPlot)) 
            Ch1AcfPlots = np.vstack((Ch1AcfPlots, acfPlotCh1))                                                   #ADDS ONTO THE GROWING LIST OF BOX ACFS
            Ch2AcfPlots = np.vstack((Ch2AcfPlots, acfPlotCh2))
            varDict = {"Signal Shift":shift,
                       "Ch1 Period":periodCh1, "Ch1 Width":widthCh1, "Ch1 Max":maxCh1, "Ch1 Min":minCh1, "Ch1 Amp":ampCh1, "Ch1 Rel Amp":relAmpCh1,
                       "Ch2 Period":periodCh2, "Ch2 Width":widthCh2, "Ch2 Max":maxCh2, "Ch2 Min":minCh2, "Ch2 Amp":ampCh2, "Ch2 Rel Amp":relAmpCh2}
            for key, var in varDict.items():        #iterates through the dictionary...
                paramDict[key].append(float(var))   #...and appends the appropriate variable into the growing lists in paramdict

        for grownArray in [Ch1AcfPlots, Ch2AcfPlots, ccfPlots]:  
            grownArray = np.delete(grownArray, obj=0, axis=0) #deletes the empty array in each of the respective correlation plot arrays
        
        listOfCFs = []
        listOfCFs.append(plotCF(ccfPlots, boxSavePath, paramDict["Signal Shift"], channel = "", cfType = "CCF"))
        for key, var in {"Ch1 Period":Ch1AcfPlots, "Ch2 Period": Ch2AcfPlots}.items():
            listOfCFs.append(plotCF(var, boxSavePath, paramDict[key], channel = key[:3])) #calls plot acf twice
        
        df = pd.DataFrame(np.hstack(listOfCFs), columns=["X Axis", "CCF Mean", "CCF Std Dev", 
                                               "X Axis", "Ch1 Mean", "Ch1 Std Dev", 
                                               "X Axis", "Ch2 Mean", "Ch2 Std Dev"])
        df.to_csv(boxSavePath / ("cfPlots.csv"))

        periodsCh1 = [x for x in paramDict["Ch1 Period"][1:] if np.isnan(x) != True]
        periodsCh2 = [x for x in paramDict["Ch2 Period"][1:] if np.isnan(x) != True]
        pcntZerosCh1 = ((numBoxes-len(periodsCh1))/numBoxes)*100
        pcntZerosCh2 = ((numBoxes-len(periodsCh2))/numBoxes)*100

        for ch in ["Ch1", "Ch2"]:
            plotPeaks(paramDict[ch + " Width"][1:], paramDict[ch + " Min"][1:], paramDict[ch + " Max"][1:], paramDict[ch + " Amp"][1:], boxSavePath, channel=ch)

        summaryDict={"Filename":nameWithoutExtension}
        summaryDict["# of Boxes"] = numBoxes
        summaryDict["Ch1 Pcnt Zero Boxes"] = pcntZerosCh1
        summaryDict["Ch2 Pcnt Zero Boxes"] = pcntZerosCh2
        if compareFiles == True:
            summaryDict["Group Name"] = groupName

        mean, median, std, sem =  calcListStats(paramDict["Signal Shift"][1:])
        for stat, val in {"Mean":mean, "Median":median, "StDev":std, "SEM":sem}.items():
            summaryDict[stat + " Signal Shift"] = val
        for ch in ["Ch1", "Ch2"]:
            for meas in ["Period", "Width", "Max", "Min", "Amp", "Rel Amp"]:
                mean, median, std, sem =  calcListStats(paramDict[ch + " " + meas][1:])
                for index, item in {1:mean, 2:median, 3:std, 4:sem}.items():
                    paramDict[ch + " " + meas].insert(index, item)
                for stat, val in {"Mean":mean, "Median":median, "StDev":std, "SEM":sem}.items():
                    summaryDict[ch + " " + stat + " " + meas] = val
        
        for key in summaryDict.keys():
            if key not in columnHeaders:
                columnHeaders.append(key)

        listOfMeasurements = []
        for finishedList in paramDict.values():
            listOfMeasurements.append(finishedList)
        saveBoxValues(listOfMeasurements, boxSavePath, columnNames)  #SINGLE FUNCTION THAT PRINTS SUMMARY AND BOX SIZE FOR EVERYTHING

        masterStatsList.append(summaryDict)
        print(str(round((i+1)/len(fileNames)*100, 1)) + "%" + " Finished with Analysis")
          











df = pd.DataFrame(masterStatsList, columns=columnHeaders)
df.to_csv(directory + "/0_fileStats.csv")

compareSavePath = pathlib.Path(directory + "/0_comparisons") #sets save path for output
compareSavePath.mkdir(exist_ok=True, parents=True)  #makes save path for output, if it doesn't already exist
comparisonsToMake = ["Mean Signal Shift"]
for ch in ["Ch1", "Ch2"]:
    for meas in ["Period", "Width", "Min", "Max", "Amp", "Rel Amp"]:
        comparisonsToMake.append(ch + " Mean " + meas)

if compareFiles == True:
    for comparison in comparisonsToMake:
        try:
            plotComparisons(df, comparison, compareSavePath)
        except ValueError:
            pass



