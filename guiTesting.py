import enum
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

np.seterr(divide='ignore', invalid='ignore')

boxSizeInPx = 20                #Desired box size for analysis
plotIndividualACFs = False      #True = plots signal trace and ACF curve for every box; False = only plots pop means. 
plotIndividualCCFs = False      #True = plots signal trace and CCF curve for every box; False = only plots pop means. 
plotIndividualPeaks = False     #True = plots signal trace and peak picking for every box; False = only plots pop statistics.
compareFiles = True            #True = generates plots comparing the different groups in your dataset; False = only writes wave stats
fileNameIndex = -1              #Necessary for "compareFiles = True", identifies the group index in the filename.
acfPeakProm = 0.1               #Minimum peak prominence to choose in an ACF, set 0-1. Larger values are more stringent. 
baseDirectory = "/Users/bementmbp/Desktop/testDatasets"      #Base directory for the GUI. Can hard code file path by commenting line 23 and uncommenting line 24. 
graphicUserInterface = True
exitButtonVar = False #variable for cancel button. set to false automatically
startButtonVar = False
       
'''*** Start GUI Window ***'''
if graphicUserInterface == True:

    '''GUI Window '''
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
    compareFilesVar = tk.BooleanVar() #variable for plotting group-wise comparisons

    #function for getting path to user's directory
    def getFolderPath():
        folderSelected = askdirectory()
        folderPath.set(folderSelected)

    #function for hitting cancel button
    def on_quit(): 
        global exitButtonVar #references the global variable
        exitButtonVar = True #sets it to true
        root.destroy() #destroys window

    def on_start(): 
        global startButtonVar #references the global variable
        startButtonVar = True #sets it to true
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

    ttk.Checkbutton(root, variable=compareFilesVar).grid(column=0, row=8, sticky='E', padx=15) #plot group-wise comparisons
    ttk.Label(root, text='Plot group-wise comparisons').grid(column=1, row=8, columnspan=2, padx=10, sticky='W')

    #Creates the 'Start Analysis' button
    startButton = ttk.Button(root, text='Start Analysis', command=on_start) #creates the button and bind it to close the window when clicked
    startButton.grid(column=1, row=9, pady=10, sticky='W') #place it in the tk window

    #Creates the 'Cancel' button
    cancelButton = ttk.Button(root, text='Cancel', command=on_quit) #creates the button and bind it to on_quit function
    cancelButton.grid(column=0, row=9, pady=10, sticky='E') #place it in the tk window
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
    compareFiles = compareFilesVar.get()

    #make dictionary of parameters for log file use
    logParams = {
        "Box Size(px)" : boxSizeInPx,
        "Base Directory" : baseDirectory,
        "ACF Peak Prominence" : acfPeakProm,
        "Group Names" : groupNames,
        "Plot Individual ACFs" : plotIndividualACFs,
        "Plot Individual CCFs" : plotIndividualCCFs,
        "Plot group-wise comparisons" : compareFiles
        }

    errors = []
    errorMessage = False
    if exitButtonVar == True:
        sys.exit('You opted to cancel the script before running')
    elif exitButtonVar == False and startButtonVar == False:
        sys.exit('You opted to cancel the script before running')
    if compareFiles == True and len(groupNames) < 2:
        errorMessage = True
        errors.append('If you want to compare multiple groups, you must enter more than one group name')
    if acfPeakProm > 1 :
        errorMessage = True
        errors.append("The ACF peak prominence can not be greater than 1, set 'ACF peak prominence threshold' to a value between 0 and 1. More realistically, a value between 0 and 0.5")
    if len(groupNames) > 1 and compareFiles == False:
        errorMessage = True
        errors.append("You entered group names, but didn't click the 'Plot group-wise comparisons' checklist")
    if len(baseDirectory) < 1 :
        errorMessage = True
        errors.append("You didn't enter a directory to analyze")


    if errorMessage == True:
        if len(errors) == 1 :
            print(errors[0])
            sys.exit("It's okay! Just fix that error and try again")
        else:
            for error in enumerate(errors):
                print(str(error[0]+1) + ') ' + error[1])
            sys.exit("It's okay! Just fix those errors and try again")

elif graphicUserInterface != True:
    logParams = {
    "Box Size(px)" : boxSizeInPx,
    "Base Directory" : baseDirectory,
    "ACF Peak Prominence"  : acfPeakProm,

    "Plot Individual Peaks": plotIndividualPeaks,
    "Plot Individual ACFs" : plotIndividualACFs,
    "Plot Individual CCFs" : plotIndividualCCFs,
    "Plot group-wise comparisons" : compareFiles
    }
    fileNames = [fname for fname in os.listdir(baseDirectory) if fname.endswith('.tif')]
    groupNames = []
    for fileName in fileNames:
        extensionless = fileName.rsplit(".",1)[0]
        group = extensionless.split("_")[fileNameIndex]
        if group not in groupNames:
            groupNames.append(group)
    logParams["Group Names"] = groupNames



'''*** End GUI Window ***'''

print('boxSizeInPx is',boxSizeInPx)
print('baseDirectory is',baseDirectory)
print('acfPeakProm is',acfPeakProm)
print('groupNames is',groupNames)
print('plotIndividualACFs is',plotIndividualACFs)
print('plotIndividualCCFs', plotIndividualCCFs)
print('compareFiles is', compareFiles)

# Assert that you have to type in group names if you set compare files equal true