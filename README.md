# signalProcessing2C.py
## Expected Output:

signalProcessing2C.py is written to batch analyze oscillatory dynamics in both 1-channel and 2-channel time lapse datasets. The primary dependencies are numpy, pandas, seaborn, scikit-image, scipy, and matplotlib. See the environment.yml or requirements.txt file to create your own environment. 

Here we will walk through the analysis of the following dataset: starfish cells treated with the drug Latrunculin B for consecutively longer periods of time. The example on the left (Group 1) is untreated, the example in the middle (Group 2) was treated for ~15 minutes, the example on the right (Group 3) was treated for ~50 minutes. Small cropped examples from each group can be found in the testDatasets folder of this repository. 

![GitHub-Mark-Light](https://github.com/zacswider/signalProcessing/blob/main/README_Images/groups_dark.gif#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/signalProcessing/blob/main/README_Images/groups_light.gif#gh-light-mode-only)

In this example, we will be processing a 2-channel dataset, however, 1-channel datasets are also accepted. If a 1-channel dataset is detected it will be processed in the same manner as a 2-channel dataset, they just won't produce a crosscorrelation, which requires two channels. 

In this workflow, each channel is broken up in n boxes (the box size will depend on the size of the features of interest).

![GitHub-Mark-Light](https://github.com/zacswider/signalProcessing/blob/main/README_Images/boxes_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/signalProcessing/blob/main/README_Images/boxes_light.jpg#gh-light-mode-only)

And each box is measured independently as follows:

The mean pixel intensity in each box, when viewed over time, is a readout for the oscillatory dynamics in that region. For each channel, the period of the oscillatory signal can be estimated by calculating the autocorrelation of that signal.

![GitHub-Mark-Light](https://github.com/zacswider/signalProcessing/blob/main/README_Images/autocorrelation_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/signalProcessing/blob/main/README_Images/autocorrelation_light.jpg#gh-light-mode-only)

For 2-channel datasets, the temporal shift (if any) between the two signals is estimated by calculating the crosscorrelation of the two channels. 

![GitHub-Mark-Light](https://github.com/zacswider/signalProcessing/blob/main/README_Images/crosscorrelation_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/signalProcessing/blob/main/README_Images/crosscorrelation_light.jpg#gh-light-mode-only)


Oscillation properties (e.g., signal peak, signal trough, signal amplitude, temporal duration) can be determined from each waveform. As a precaution for noisy data, which real-world data are more often than not, the signal are smoothed using a Savitzky–Golay filter to avoid quantifying spurious peaks.

![GitHub-Mark-Light](https://github.com/zacswider/signalProcessing/blob/main/README_Images/peaks_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/signalProcessing/blob/main/README_Images/peaks_light.jpg#gh-light-mode-only)

Once each box has been independently quantified, they can be combined to estimate properties of the wave population. For example, in the example above we measured a period of 12 frames, is that measurement representative of the whole sample? By looking at the distribution of all period measurements, we can see that it is. 

![GitHub-Mark-Light](https://github.com/zacswider/signalProcessing/blob/main/README_Images/meanACF_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/signalProcessing/blob/main/README_Images/meanACF_light.jpg#gh-light-mode-only)

Similarly, we can assess the population of signal shift measurements, and oscillation/wave properties.

![GitHub-Mark-Light](https://github.com/zacswider/signalProcessing/blob/main/README_Images/meanPeaks_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/signalProcessing/blob/main/README_Images/meanPeaks_light.jpg#gh-light-mode-only)

Finally, if we choose to compare different groups, we will get file full of plots comparing each signal metric between groups.

![GitHub-Mark-Light](https://github.com/zacswider/signalProcessing/blob/main/README_Images/comparisons_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/signalProcessing/blob/main/README_Images/comparisons_light.jpg#gh-light-mode-only)


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
2) If you haven't restarted the terminal since installing your environment, you're already in the correct directory. If you aren't sure, type `cd` into the terminal and hit enter. Next, type `cd Desktop/signalProcessing-main` into the terminal and hit enter to navigate to the unzipped flder.
3) Activate the newly installed environment by typing `conda activate waves` into the terminal and hitting enter. 
4) Type `python3 signalprocessing2c.py` into the terminal and hit enter to run the script.
5) It may take a second to connect to the correct environment the first time you run, but next a window will appear asking you for some variable to enter:
![alt text](https://github.com/zacswider/signalProcessing/blob/main/README_Images/GUI_small.jpg)









