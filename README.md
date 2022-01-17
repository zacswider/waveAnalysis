# signalProcessing scripts
These tools were developed for analyzing excitable / oscillatory dynamics in living cells. The 2-channel analysis script signalProcessing2C.py is conceptually based off of a MATLAB framework written by Marcin Leda and Andrew Goryachev (published in Bement _et al.,_ 2015; PMID 26479320), and was reimagined here by Zac Swider and Ani Michaud in Python form to increase speed, accuracy, and access. This script analyzes wave period, amplitudes, width, and (if applicable) the temporal shift between signals in short time lapse datasets (tens of frames, typically, see the repository for more details). The 1-channel rolling script extends upon this functionality by analyzing these metrics across extended timelapse datasets (hundreds - thousands of frames, see the repository for more details). The assemblyDisassembly plots script was written to analyze the contributions of wave assembly and disassembly to the overall wave population in the cell (see the repository for more details).

## Preparing data for analysis
Before running any analysis on your 1 or 2-channel time lapse datasets, be sure to complete all necessary pre-processing steps. Some thing to consider:
- Any significant two-dimensional drift in your data will alter the detected wave dynamics. If drift is detectable, register your data ahead of time.
- Black spaces (e.g. from drift correction, or true background) should be cropped out. The period detection functions will not return a value if no period is detected in a dark spaces, but the amplitude functions won't know the difference.
- Bleaching or z-drift will both affect amplitude and width measurements. The best approach is to not have bleaching/drift to begin with as bleach correction algorithms can introduce their own artifacts. However, if desired, correct your data for bleaching before analyzing.

## Downloading the scripts and setting up Python on your computer

In this section I will assume that you have no idea what Python is, or how to use it. If you already know how to use Python and Conda, you can probably skip this section. 

### On a Mac:

1) Click the big green "Code" button in the upper right corner of this repository. Select "Download ZIP", and then unzip it to your desktop.
2) Go to https://www.anaconda.com/products/individual and download/install Anaconda (a distribution of Python and a package manager). 
3) Once installed, open the terminal by pushing `Command–Space bar`, typing "terminal", and hitting enter.
4) Type `cd Desktop/signalProcessing-main` into the terminal and hit enter to navigate to the unzipped flder.
5) Type `conda env create --file environment.yml` into the terminal and hit enter. This will install a bunch of scientific computing/analysis packages into an environment call "waves". The script will need the packages in this environment to run correctly.
6) When complete, the final lines in your terminal should say 
```
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate waves
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

### On a PC:

1) Please check back shortly, I am in the process of updating this readme

## Running the scripts 

1) If it's not already open, open the terminal by pushing `Command–Space bar`, typing "terminal", and hitting enter.
2) If you haven't restarted the terminal since installing your environment, you're already in the correct directory. If you aren't sure, type `pwd` into the terminal and hit enter. Next, type `cd ~/Desktop/signalProcessing-main` into the terminal and hit enter to navigate to the unzipped flder.
3) Activate the newly installed environment by typing `conda activate waves` into the terminal and hitting enter. 
4) Type `python3 signalprocessing2c.py` into the terminal and hit enter to run the script.
5) It may take a second to connect to the correct environment the first time you run, but next a window will appear asking you for some parameters to adjust:

<img width="496" alt="Screen Shot 2022-01-17 at 8 51 59 AM" src="https://user-images.githubusercontent.com/32859488/149791989-d627fab0-c64d-4690-923b-fb56bbe7cb7f.png">

### Setting parameters:
1) This is the source directory for your analysis. Navigate to it using the "Select source directory button". This directory should have one or more 1 or 2-channel time lapse datasets saved in standard standard `tzcyx` order. 
2) This is the box size used for analysis. Boxes should be large enough to filter out noise, but small enough that they don't over-fill the structures being analyzed and merge multiple wave signals together. A good way to empirically find the apppropriate box size is to open your data in [FIJI](https://imagej.net/software/fiji/), draw a box with the rectangle selection tool, open up the z-axis profile plotter `Image > Stacks > Plot Z-axis Profile`, click the "Live" button, and adjust the box dimensions to find a size that you feel like accurately captures the temporal dynamics. Generally, box size should be roughly equal to spatial wave width. 
3) The is the minimum prominence in the autocorrelation curve to be considered a genuine period. Use the default parameter `0.1` to begin, but if data is noisy, the threshold can be increase to filter out artificial noise peaks. The maximum value for this field is 1. 
4) If you want to compare the population measurements between different groups, enter the groups names in this space. These names *must* be present within the names of the file being processed. 
5) If you check this box, a graphical output of the autocorrelation for every box analyzed will be saved to the analysis folder.
6) If you check this box, a graphical output of the crosscorrelation for every box analyzed will be saved to the analysis folder.
7) If you check this box, a graphical output of the wave peak analalysis for every box analyzed will be saved to the analysis folder.

That's it! Just click start analysis, and the script will process and save the output to your analysis folder. 





