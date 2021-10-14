from genericpath import exists
from tkinter.filedialog import askdirectory 	#Imports the ability of open a GUI asking for the base folder to start working from
import pathlib                                  #Object-oriented filesystem paths
import numpy as np                              
import pandas as pd                             
import seaborn as sns                           
import skimage.io as skio                       
import matplotlib.pyplot as plt                 
import scipy.signal as sig                      
import os                                       
import sys                                      
import math         
import fnmatch                                  #filename matching 
np.seterr(divide='ignore', invalid='ignore')

boxSizeInPx = 20                #ENTER DESIRED BOXED SIZE HERE
plotIndividualACFs = True      #TRUE = PLOTS BOXES; FALSE = ONLY PLOTS POP MEANS
plotIndividualCCFs = False      #not functioning
plotIndividualPeaks = False      #TRUE = plots peaks for each box. 
smoothACF = False               #TRUE = smooths the ACF to better eliminate noisy peaks
smoothMySignal = True           #TRUE = SMOOTHS THE BOX MEANS PRIOR TO CALCULATING THE WAVE AMPLITUDE AND WIDTHS
acfPeakProm = 0.1               #prominance value for peak detection
baseDirectory = "/Users/aniv/Desktop/test-cropped/"         #BASE DIRECTORY FOR THE GUI

def findWorkspace(directory, prompt):               #GUI for selecting your working directory
    #targetWorkspace = askdirectory(initialdir=directory, message=prompt)                           #opens prompt asking for folder, keep commented to default to baseDirectory
    targetWorkspace = directory                                                                     #comment this out later if you want a GUI
    filelist = [fname for fname in sorted(os.listdir(targetWorkspace)) if fname.endswith('.tif')]   #Makes a list of file names that end with .tif
    return(targetWorkspace, filelist)                                                               #returns the folder path and list of file names

def findBoxMeans(imageArray, boxSize):              #Finds the means for each box. Accepts an image array as a parameter, as well as the desired box size
    depth = imageArray.shape[0]                     #number of frames
    yDims = imageArray.shape[1]                     #number of pixels on y-axis
    xDims = imageArray.shape[2]                     #number of pixels on x-axis
    yBoxes = yDims // boxSize                       #returns int result of floor division; number of boxes on the y axis
    xBoxes = xDims // boxSize                       #returns int result of floor division; number of boxes on the x axis
    growingArray = np.zeros((1, depth))             #makes a starting array of 64 bit zeros that can be added onto later. shape = (1, depth of imageStack)

    for x in range(xBoxes):                         #iterates through the number of boxes on the x-axis
        for y in range (yBoxes):                    #iterates through the number of boxes on the y-axis
            boxMean = np.array([np.mean(imageArray[:,(y*boxSize):(y*boxSize+boxSize),(x*boxSize):(x*boxSize+boxSize)], (1,2))])  #creates a 2d array of shape (depth, 1) containing the mean values of the px within the box for each slices
            growingArray = np.append(growingArray, boxMean, axis = 0)      #appends the 2d array onto the growing array
    growingArray = np.delete(growingArray, 0, axis=0)                      #deletes 0's used for array initialization
    return(growingArray)                                                   #returns ndarray of shape (number of boxes, number of frames)

def findCCF(ch1BoxMeans, ch2BoxMeans, npts, nBoxes): #Finds the CCF for all boxes when image has 2 channels
    npts = ch1BoxMeans.shape[1]                      #find the number of frames in the image
    ccfArray = np.zeros((npts*2-1))                  #initialize a 1D array for the master ccf output
    shifts = []                                      #empty list which will be filled with shift values later
    

    for i in range(0, nBoxes): #iterates through the number of boxes
        ccov = np.correlate(ch1BoxMeans[i] - ch1BoxMeans[i].mean(), ch2BoxMeans[i] - ch2BoxMeans[i].mean(), mode='full')    #compute full cross correlation between ch1 and ch2
        ccf = ccov / (npts * ch1BoxMeans[i].std() * ch2BoxMeans[i].std())                                                   #normalizes the crosscorr from -1 to +1 
        
        peaks, dict = sig.find_peaks(ccf)             #Find the peaks from the ccf. ndarray with location of local maxima. 
        peaksDiff = abs(peaks - ccf.shape[0]//2)      #ndarray with the absolute difference between each peak and the middle value of the ccor array
        delay = np.argmin(peaksDiff)                  #index array reporting difference between the first peak and the minimum value (zero)
        delayIndex = peaks[delay]                     #gets index of delay value inside peaks array
        actualShift = delayIndex - ccf.shape[0]//2    #actual shift value of first maxima minus middle of ccor
        
        ccfArray = np.vstack((ccfArray, ccf))    #appends the ccf for each box to the master list as a new row
        shifts.append(actualShift.item())        #appends the shift for each box to 'shifts'

        if plotIndividualCCFs == True:  #this will get filled in once I write it
            pass ## fill in later
    
    ccfArray = np.delete(ccfArray, obj=0, axis=0)  #deletes the 0's used to initialize the array
    return(ccfArray, shifts)                       #return the array of ccfs, as well as the shifts

def findACF(chBoxMeans, npts, nBoxes):      #Finds the ACF for all boxes
    lags = np.arange(-npts + 1, npts)       #creates an array from -npts to +npts. Number of frames in the correlation
    acorArrays = np.zeros(npts*2-1)         #initialize a 1D array for the master acf output
    periods = np.zeros(nBoxes)              #initialize an array for the periods for each box
    
    for channel in chBoxMeans:              #iterates through each slice of chBoxMeans (the array containing all the means for each box for all channels)
        lagList = []                        #empty list which will be filled with period values later
        acorArray = np.zeros(npts*2-1)      #temporary array for the acf output of all boxes for an individual channel
        for i in range(0, nBoxes):          #iterates through the boxes
            acov = np.correlate(channel[i] - channel[i].mean(), channel[i] - channel[i].mean(), mode='full')    #compute full autocorrelation
            acor = acov / (npts * channel[i].std() ** 2)                                #normalizes the crosscorr from -1 to +1      
            peaks, dict = sig.find_peaks(acor, prominence=acfPeakProm)                  #Find the peaks from the acf. ndarray with location of local maxima. 
            peaksDiff = abs(peaks - acor.shape[0]//2)       #ndarray with the absolute difference between each peak and the middle value of the ccor array

            try:                                                        
                delay = np.min(peaksDiff[np.nonzero(peaksDiff)]) #try to find the difference between the first peak and zero
            except ValueError:
                delay = np.nan                                   #reports nan if no suitable peak exist
            acorArray = np.vstack((acorArray, acor))             #appends the acf for each box to the channel list (acorArray)
            lagList.append(delay)                                #appends the shift for each box to 'lagList'
        
        acorArray = np.delete(acorArray, 0, axis = 0)       #deletes 0's from acorArray initialization
        acorArrays = np.vstack((acorArrays, acorArray))     #stacks the acorArray for each channel into the master list vertically
        periods = np.vstack((periods, lagList))             #makes 2D array of periods [[Ch1], [Ch2]]

    periods = np.delete(periods, 0, axis=0)                     #deletes initializing row of 0's in period list
    acorArrays = np.delete(acorArrays, 0, axis=0)               #removes 0s from master acorArrays initialization
    
    if chBoxMeans.shape[0] == 2:                                            #if there are 2 channels...
        acorArrays = np.reshape(acorArrays, (2, nBoxes, npts*2-1))    #reshape the final array into 3D, where each channel is a z-slice
        return(acorArrays, periods)                                                  #return the reshaped array
    else:                                                                   #otherwise, just return the array as-is (1-channel)
        return(acorArrays, periods)

def smoothWithSavgol(signal, windowSize, polynomial): #smooths noisy signals
    smoothedSignal = sig.savgol_filter(signal, windowSize, polynomial)
    return smoothedSignal

def analyzePeaks(chBoxMeans, nBoxes, boxSavePath, npts): #find and analyze signal peaks for each channel. Plot individual peaks if desired
    finalArray = np.zeros(5)          #initialize a master array for the output of the function. 5 elements (width, max, min, amp, relamp)
    
    channelNumber = 1                 #counter for naming channels 
    for channel in chBoxMeans:        #for each channel in the boxMeans array...
        tempArray = np.zeros(5)       #temporary array for the output from each channel
        for i in range(0, nBoxes):    #for each box...
            if smoothMySignal == True:                              #if true, smooth the signal
                smoothed = smoothWithSavgol(channel[i], 11, 2)      
                minVal = np.min(channel[i])                         #find the minimum
                maxVal = np.max(channel[i])                         #find the maximum
                xAxis = np.arange(len(channel[i]))                  #define the xAxis as the length of the correlation (I think redundant with npts)
                smoothPeaks, smoothedDicts = sig.find_peaks(smoothed, prominence=(maxVal-minVal)*0.1)  #Find peaks in the smoothed data. DETECTED NO PEAKS IN DECAYING DATASET
                
                if len(smoothPeaks) > 0:        #if there are peaks to be found....
                    proms, leftBase, rightBase = sig.peak_prominences(smoothed, smoothPeaks)    #find and store info on the peak prominances, and left/right bounds
                    widths, heights, leftIndex, rightIndex = sig.peak_widths(smoothed, smoothPeaks, rel_height=0.5) #returns [0]=widths, [1]=heights, [2]=left bound, [3]=right bound (all ndarrays)
                    if plotIndividualPeaks == True:                             #if this is set to true, it will make a graph for each box's peaks
                        savePath = os.path.join(boxSavePath, "Peak_Plots")      #where to save the graphs
                        os.makedirs(savePath, exist_ok=True)                    
                        printBoxPeaks(channelNumber, channel[i], i, smoothed, smoothPeaks, heights, leftIndex, rightIndex, proms, savePath, nBoxes, npts)  #calls the function that graphs/saves the individual peaks
                        
                    width = np.mean(widths, axis=0)                     #calculate mean width of all peaks in box
                    max = np.mean(smoothed[smoothPeaks], axis=0)        #calculate mean max of all peaks in box
                    min = np.mean(smoothed[smoothPeaks]-proms, axis=0)  #calculate mean min of all peaks in box
                    amp = max-min                                       #calculate amp from max and min for all peaks in box
                    relAmp = amp/min                                    #calculate relative amplitude for all peaks in box
                
                else:   #if there are no peaks, return 'NaN'
                    width = np.NaN
                    max = np.NaN
                    min = np.NaN
                    amp = np.NaN
                    relAmp = np.NaN
            
            else:
                sys.exit("Keep 'smoothMySignal =True' for now") #Idk why this is hard-coded this way.
            
            peakparams = np.array((width, max, min, amp, relAmp))   #for each box, make an array of the average width, max, min, amp, relAmp
            tempArray = np.vstack((tempArray, peakparams))          #stack the peakparams array into the tempArray for the current channel
        channelNumber = channelNumber + 1                           #iterate the counter to the next channel
        
        
        tempArray = np.delete(tempArray, obj=0, axis=0)             #delete the initializing 0's in the tempArray
        finalArray = np.vstack((finalArray, tempArray))             #stack the tempArray (for a channel) into the master finalArray (for all channels)
        
    finalArray = np.delete(finalArray, obj=0, axis=0)               #delete the initializing 0's from the master finalArray
    
    if chBoxMeans.shape[0] == 2:                                            #if there are 2 channels...
        finalArray = np.reshape(finalArray, (2, chBoxMeans.shape[1], 5))    #reshape the final array into 3D, where each channel is a z-slice
        return(finalArray)                                                  #return the reshaped array
    else:                                                                   #otherwise, just return the array as-is (1-channel)
        return(finalArray)

def plotACFs(chBoxMeans, acorArrays, periods, npts, nBoxes, boxSavePath, subFolderName): #function for printing individual box ACFs 
    acfSavePath = os.path.join(boxSavePath, subFolderName)
    os.makedirs(acfSavePath, exist_ok=True)

    plotProfileAxis = np.arange(npts)
    acfAxis = np.arange(-npts + 1, npts) 
    periods = periods.T
    
    for i in range(0, nBoxes):
        fig, axs = plt.subplots(nrows=2)
        fig.subplots_adjust(hspace=0.4)
        ax = axs[0]
        ax.plot(plotProfileAxis, chBoxMeans[i]) #plot the mean signal for that box
        ax.set_ylabel('Mean box px value')
        ax.set_xlabel('Time (frames)')
        
        
        if periods[i] == 0:
            ax = axs[1]    
            ax.set_ylabel('auto-correlation')
            ax.plot(acfAxis, acorArrays[i]) #plot acf for that box   
            ax.set_xlabel("Periodic signal not detected")
            graphName = "box" + str(i) + ".png"
        
        else:
            ax = axs[1]    
            ax.set_ylabel('auto-correlation')
            ax.plot(acfAxis, acorArrays[i]) #plot acf for that box 
            ax.set_xlabel("Period is " + str(periods[i]) + " frames")
            plt.axvline(x=periods[i], alpha = 0.5, c = 'red', linestyle = '--')
            plt.axvline(x=-periods[i], alpha = 0.5, c = 'red', linestyle = '--')
            graphName = "box" + str(i) + ".png"

        boxName = os.path.join(acfSavePath, graphName)
        print("Saving Graph ", graphName)
        plt.savefig(boxName, dpi=75, )
        plt.clf()
        plt.close(fig)

def printBoxPeaks(channelNumber, raw, boxNumber, smoothed, smoothPeaks, heights, leftIndex, rightIndex, proms, savePath, nBoxes, npts): #graphs/saves individual peak plots
    x = np.arange(npts)                                         #make array of numbers from 0 to npts
    fig, axs = plt.subplots(nrows=2)                            #initialize a figure with 2 subplots
    fig.subplots_adjust(hspace=0.4)                             
    ax = axs[0]                                                 #switch to the first plot 
    ax.plot(x, raw, color='tab:blue', label='raw')              #plot the raw box signal
    ax.plot(x,smoothed, color='tab:orange', label='smoothed')   #plot the smoothed signal over the raw
    ax.legend(loc='upper right', fontsize='small', ncol=1)      
    ax.set_ylabel('Mean box px value')                          
    ax.set_xlabel('Time (frames)')    
    ax=axs[1]                                                   #switch to the second plot
    ax.plot(x,smoothed, color='tab:orange', label='smoothed')   #plot the smoothed signal again

    for i in range(smoothPeaks.shape[0]):                       #for each entry in smoothPeaks, plot the amp, and FWHM lines
        ax.hlines(heights[i], leftIndex[i], rightIndex[i], color='tab:blue', alpha = 1, linestyle = '-') 
        ax.vlines(smoothPeaks[i], smoothed[smoothPeaks[i]]-proms[i], smoothed[smoothPeaks[i]], color='tab:purple', alpha = 1, linestyle = '-') 
    
    ax.hlines(heights[0], leftIndex[0], rightIndex[0], color='tab:blue', alpha = 1, linestyle = '-', label='FWHM')#
    ax.vlines(smoothPeaks[0], smoothed[smoothPeaks[0]]-proms[0], smoothed[smoothPeaks[0]], color='tab:purple', alpha = 1, linestyle = '-', label='Peak amplitude')
    ax.legend(loc='upper right', fontsize='small', ncol=1)  
    ax.set_ylabel('Mean box px value')
    ax.set_xlabel('Time (frames)')    
    
    graphName = "Ch" + str(channelNumber) + "_BoxNo" + str(boxNumber) + ".png" #sets graph name
    print("Saving Graph ", graphName)
    plt.savefig(os.path.join(savePath, graphName), dpi=74)                     #saves the figure  
    plt.clf()
    plt.close(fig)

def plotCF(corFunction, npts, shifts, fileName): #plot the correlation function (CCF or ACF)
    plotSavePath = os.path.join(boxSavePath, fileName) 
    name = fileName.split('.')[0]  #gets the name of the future file ('CCF' or 'ACF' without the .png suffix)

    mean = np.mean(corFunction, axis=0) #calculate the mean of the correlation function across boxes for each timepoint
    std = np.std(corFunction, axis=0)   #calculate the std of the correlation function over time
    frames = np.arange(-npts + 1, npts) #array for the x-axis of the plot
    cfData = np.vstack((frames, mean, std)).T   #stacks the frames, mean, and std values into columns

    columnNames = ["Frames", name+" Mean", name+" Std"]    #names the column names with either 'ACF' or 'CCF' prefix
    cfDataDf = pd.DataFrame(cfData, columns=columnNames)   #create a dataframe of the data

    shifts = [x for x in shifts if np.isnan(x) !=True]     #filters nans from the shifts/periods before plotting
    
    plt.subplot(2,1,1)  #top subplot                                
    plt.subplots_adjust(wspace=0.4) 
    plt.subplots_adjust(hspace=0.4) 
    plt.plot(frames, mean)                                    #plot the frames and mean correlation function
    plt.fill_between(frames, mean-std, mean+std, alpha = 0.5) #plot the std and fill between the bounds
    plt.title("Average " + name + " curve ± Std Dev")            
    plt.xlabel("Time (Frames)") 
    
    plt.subplot(2,2,3)  #bottom left subplot
    plt.hist(shifts)    #plot the shifts or periods
    
    if fnmatch.fnmatch(name, '*ACF*'):              #if the name includes 'ACF', label the axes as noted below
        plt.xlabel("Histogram of Period values")                                 
    if fnmatch.fnmatch(name, '*CCF*'):              #if the name includes 'CCF', lable the axes as noted below
        plt.xlabel("Histogram of Shift values")
    plt.ylabel("Occurrences")  

    plt.subplot(2,2,4)                              #bottom right subplot
    plt.boxplot(shifts)                             #boxplot of shift/period values
    if fnmatch.fnmatch(name, '*ACF*'):              #if the name includes 'ACF', label the axes as noted below
        plt.xlabel("Boxplot of Period values")      
        plt.ylabel("Measured Period (frames)")
    if fnmatch.fnmatch(name, '*CCF*'):              #if the name includes 'CCF', lable the axes as noted below
        plt.xlabel("Boxplot of Shift values")
        plt.ylabel("Measured Shift (frames)")   
    plt.xticks(ticks=[])                

    plt.savefig(plotSavePath, dpi=80)              #saves the figure
    plt.close()                                    #clears the figure
    
    return(cfDataDf)

def mergeAndSave(Df1, Df2, savePath, save=True, fileName=" ", **kwargs):    #merge and save dataframes
    mergedDf = pd.merge(Df1, Df2)   #merges two dataframes with default options
    
    if save==True:                                          #if you specified that you want to save the dataframe...
        fullSavePath = os.path.join(boxSavePath, fileName)  #creates the save path
        mergedDf.to_csv(fullSavePath, index=False)          #writes the dataframe to a csv file, ignoring indices
        return(mergedDf)
    else:
        return(mergedDf)
    
def calculatePcntZeros(array):    #calculate the percent zeros (nan's) in an array
    zeros = np.count_nonzero(np.isnan(array), axis=1)   #finds all 'nan' values along a row
    pcntZeros = ((zeros/periods.shape[1])*100)          #calculates percent out of total number of columns
    return(pcntZeros)                                   

def plotPeaks(peaksDf, boxSavePath, fileName):          #plot all of the peak values
    savePath = os.path.join(boxSavePath, fileName)
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))                  #create the plot design
    fig.subplots_adjust(wspace=0.4)
    fig.subplots_adjust(wspace=0.4)
    ax1.hist(peaksDf.iloc[:,3], bins=20, color="tab:purple", label = "amp", alpha = 0.75)   #plot a histogram of amplitude values from the peaksDf dataframe
    ax1.hist(peaksDf.iloc[:,2], bins=20, color="tab:orange", label = "min", alpha = 0.75)   #plot a histogram of minimum values
    ax1.hist(peaksDf.iloc[:,1], bins=20, color="tab:blue", label = "max", alpha = 0.75)     #plot histogram of maximum values
    ax1.legend(loc='upper right', fontsize='small', ncol=1) 
    ax1.set_xlabel("Histogram of peak values")
    ax1.set_ylabel("Occurrences")
    
    labels = ["amp", "min", "max"]                                                  
    colors = ['tab:purple', 'tab:orange', 'tab:blue']
    plotThis = [peaksDf.iloc[:,3], peaksDf.iloc[:,2], peaksDf.iloc[:,1]]        #list of amps, mins, maxs for boxplot
    bplot = ax2.boxplot(plotThis, vert=True, patch_artist=True, labels=labels)  #create boxplot from the list of values
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_xlabel("Boxplot of peak values")
    ax2.set_ylabel("Pixel value (AU)")

    ax3.hist(peaksDf.iloc[:,0], bins=20, color="tab:blue", label = "max", alpha = 0.75) #plot a histogram of temporal width values. why max label?
    ax3.set_xlabel("Histogram of temporal width values")
    ax3.set_ylabel("Occurrences")

    plt.savefig(savePath, dpi=80)    #saves the figure
    plt.close()

def calcListStats(Df, x):   #calculate the statistics for a given range in a dataframe
    columnList = list(Df.columns[1:x])  #list of column headers for a specified range
    headers =[] #empty list for future column headers
    stats =[]   #empty list for future stats output
    for column in columnList:
        headers.extend([column+' Mean', column+' Median', column+' Std', column+' Sem'])    #for each category, create mean, median, std, sem columns
        mean = Df[column].mean()            #calculate the mean of the column
        median = Df[column].median()        #calculate median
        std = Df[column].std()              #calculate std
        sem = std/math.sqrt(Df.shape[0])    #calculate sem

        stats.extend([mean, median, std, sem])  #append this list to the stats master list

    
    statsDf = pd.DataFrame([stats], columns=headers) #create a new dataframe with all of the stats output

    return(statsDf)


### MAIN ####
directory, fileNames = findWorkspace(baseDirectory, "PLEASE SELECT YOUR SOURCE WORKSPACE")  #string object describing the file path, list object containing all file names ending with .tif
masterStatsDf = pd.DataFrame()  #empty dataframe for final stats output of all movies

for i in range(len(fileNames)):  #iterates through the .tif files in the specified directory

    print("Starting to work on " + fileNames[i] + "!")
    imageStack=skio.imread(directory + "/" + fileNames[i])                                #reads image as ndArray
    nameWithoutExtension = fileNames[i].rsplit(".",1)[0]                                  #gets the filename without the .tif extension
    boxSavePath = pathlib.Path(directory + "/0_signalProcessing/" + nameWithoutExtension) #sets save path for output
    boxSavePath.mkdir(exist_ok=True, parents=True)                                        #makes save path for output, if it doesn't already exist
    
    """Attempt the verify the number of channels in the image"""
    if imageStack.shape[1] == 2:    #imageStack.shape[1] will either be the number of channels, or the number of pixels on the y-axis
        imageChannels = 2          
    elif imageStack.ndim == 3:  #imageStack.ndim == 3 = the number of dimensions. a 1-channel stack won't have the 4th channel dimension
        imageChannels = 1
    else:
        sys.exit("Are you sure you have a standard sized image with one or two channels?")

    if imageChannels ==1:
        print("starting 1-channel workflow")
        ch1BoxMeans = findBoxMeans(imageStack, boxSizeInPx)        #ndarray of shape (# boxes, # frames)
        ch2BoxMeans=np.zeros_like((ch1BoxMeans))                  #dummy array simulating a second channel

    elif imageChannels == 2:
        print("starting 2-channel workflow")
        subs = np.split(imageStack, 2, 1) #List object containing two arrays corresponding to the two channels of the imag
        ch1 = np.squeeze(subs[0],axis=1)  #array object corresponding to channel one of imageStack. Also deletes axis 1, t
        ch2 = np.squeeze(subs[1],axis=1)  #array object corresponding to channel two of imageStack. Also deletes axis 1, t
        ch1BoxMeans = findBoxMeans(ch1, boxSizeInPx) #ndarray of shape (# boxes, # frames)
        ch2BoxMeans = findBoxMeans(ch2, boxSizeInPx) #ndarray of shape (# boxes, # frames)

    else: print("I can only handle up to 2 channels!")

    assert ch2BoxMeans.size == ch1BoxMeans.size, "ch1BoxMeans and ch2BoxMeans are not the same size, something went horribly wrong"
        
    npts = ch1BoxMeans.shape[1]     #number of frames
    nBoxes = ch1BoxMeans.shape[0]   #number of boxes

    if imageChannels ==2:
        ccfArray, shifts= findCCF(ch1BoxMeans, ch2BoxMeans, npts, nBoxes) #get ccf and shifts for each box. Each row in ccfArray is the ccf for a box over time   
        meanCCFDf = plotCF(ccfArray, npts, shifts, "CCF.png")             #plot the CCF function
    
    chBoxMeans = np.array([ch1BoxMeans, ch2BoxMeans])   #3D array of channel means (2ch, nBoxes, npts)
    del [[ch1BoxMeans, ch2BoxMeans]]

    '''Find the ACFs for each box, for all channels'''
    acorArrays, periods= findACF(chBoxMeans, npts, nBoxes) #acorArrays = ndarray where each row is the ACF for a box over time. Periods = ndarray of periods for all boxes  
    finalArray = analyzePeaks(chBoxMeans, nBoxes, boxSavePath, npts) #ch1PeakValues = ndarray of peak metrics (width, min, max, amp, relAmp) for each box
    ch1PeakValues = pd.DataFrame(finalArray[0, :, :], columns=['Ch1 Widths', 'Ch1 Maxs', 'Ch1 Mins', 'Ch1 Amps', 'Ch1 RelAmps']) #slices out Channel1 values and makes dataframe
    ch2PeakValues = pd.DataFrame(finalArray[1, :, :], columns=['Ch2 Widths', 'Ch2 Maxs', 'Ch2 Mins', 'Ch2 Amps', 'Ch2 RelAmps']) #slices out Channel2 values and makes dataframe
    
    '''Plot individual ACFs for each box if True'''
    if plotIndividualACFs == True: #this is general enough to work for any # channels. probably should turn into a function and use for 1ch/2ch
        for channel in range(0, imageChannels): 
            inputMeans = chBoxMeans[channel,:,:]                     #select the input means for the channel from the chBoxMeans
            chacorArray = acorArrays[channel,:,:]                    #select autocorrelation arrays for each channel
            chperiods = periods[channel]                             #select the periods from each channel
            subFolderName = "Ch"+str(channel+1)+"ACF_Plots"          #create a subfolder for the output
            plotACFs(inputMeans, chacorArray, chperiods, npts, nBoxes, boxSavePath, subFolderName)  #plot the ACFs for each box, for each channel
        
    '''Plot the average ACF for Ch1'''
    meanCh1ACFDf = plotCF(acorArrays[0, :,:], npts, periods[0], 'Ch1ACF.png')       #plot the mean Ch1 ACF
    meanCh1ACFDf.to_csv(os.path.join(boxSavePath,"correlations.csv"), index=False)  #save the mean Ch1 ACF as a csv
    plotPeaks(ch1PeakValues, boxSavePath, "Ch1MeanPeakMeasurements.png")      #plot the ch1PeakValues and save the graphs
    

    '''Plot Ch2 ACF (if applicable) and set up BoxMeasurements.csv'''
    if imageChannels == 2:
        meanCh2ACFDf = plotCF(acorArrays[1, :,:], npts, periods[1], 'Ch2ACF.png')     #plot the Ch2 ACF function
        plotPeaks(ch2PeakValues, boxSavePath, "Ch2MeanPeakMeasurements.png")          #plot Ch2 peak values

        boxMeasurements = pd.DataFrame({    #create a dataframe for measurements from individual boxes
        'Box#': np.arange(0,nBoxes),
        'Signal Shift' : shifts,
        'Ch1 Periods': periods[0],
        'Ch2 Periods': periods[1]
        })
        peakValues = pd.merge(ch1PeakValues, ch2PeakValues, left_index=True, right_index=True)  #merge all peak values together 
        statsDf = calcListStats(boxMeasurements, 3) #calculates statistics on the boxMeasurements over columns 1 and 2
    
    else: #create the dataframe entry for 1-channel data (no CCF or Ch2 values)
        boxMeasurements = pd.DataFrame({  #create a dataframe for measurements from individual boxes
        'Box#': np.arange(0,nBoxes),      #include box#
        'Ch1 Periods': periods[0],        #include periods
        })
        
        peakValues = ch1PeakValues #if 1-channel, just renames this df 
        statsDf = calcListStats(boxMeasurements, 2) #calculates statistics on the boxMeasurements over columns 1 only
    
    '''Adding stats values to BoxMeasurements.csv'''
    pcntZeros = calculatePcntZeros(periods)  #calculate %0's in the periods
    boxMeasurements = pd.merge(boxMeasurements, peakValues, left_index=True, right_index=True)           #add the widths, mins, maxs, amps and relAmps from each box to the dataframe
    boxMeasurements[''] = np.NaN                                                                         #adds extra column of space        
    boxMeasurements = pd.merge(boxMeasurements, statsDf, how='outer', left_index=True, right_index=True) #merge the stats calculations into the boxMeasurements dataframe
    boxMeasurements.to_csv(os.path.join(boxSavePath, "BoxMeasurements.csv"), index=False)                #save the dataframe

    '''Create entry for 0_filestats.csv (master file)'''
    masterStatsEntry = pd.DataFrame({      #dataframe entry to be added to the master stats csv file containing all movies (at end of script)
        'Filename' : nameWithoutExtension,              
        'Num Boxes' : nBoxes,
        'Ch1 Pcnt Zeros': pcntZeros[0]}, index=[0],)

    if imageChannels ==2:
        masterStatsEntry['Ch2 Pcnt Zeros'] = pcntZeros[1] #if 2-channel, add column for ch2 percent 0's 
    
    masterStatsEntry = pd.concat([masterStatsEntry, statsDf], axis=1)                #adds stats data to the master entry
    masterStatsDf = masterStatsDf.append(masterStatsEntry, ignore_index=True)        #adds the entry to the master stats file (0_filestats.csv)
    print(str(round((i+1)/len(fileNames)*100, 1)) + "%" + " Finished with Analysis") #updates progress to terminal window
    
    masterStatsDf.to_csv(os.path.join(directory, '0_filestats.csv'), index=False)    #saves master stats file at the end of the analysis 
<<<<<<< HEAD:signal-processing_ani

    
=======
    
    
>>>>>>> 8897cb7f7b1111a777ac758e201b3d34ec2064db:signal-processing_ani.py
