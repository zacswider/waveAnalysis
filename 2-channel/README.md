## Expected Output:

signalProcessing2C.py is written to batch analyze oscillatory dynamics in both 1-channel and 2-channel time lapse datasets. The primary dependencies are numpy, pandas, seaborn, scikit-image, scipy, and matplotlib. See the environment.yml or requirements.txt file to create your own environment. 

Here we will walk through the analysis of the following dataset: starfish cells treated with the drug Latrunculin B for consecutively longer periods of time. The example on the left (Group 1) is untreated, the example in the middle (Group 2) was treated for ~15 minutes, the example on the right (Group 3) was treated for ~50 minutes. Small cropped examples from each group can be found in the testDatasets folder of this repository. 

![GitHub-Mark-Light](https://github.com/zacswider/README_Images/blob/main/groups_dark.gif#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/groups_light.gif#gh-light-mode-only)

In this example, we will be processing a 2-channel dataset, however, 1-channel datasets are also accepted. If a 1-channel dataset is detected it will be processed in the same manner as a 2-channel dataset, they just won't produce a crosscorrelation, which requires two channels. 

In this workflow, each channel is broken up in n boxes (the box size will depend on the size of the features of interest).

![GitHub-Mark-Light](https://github.com/zacswider/README_Images/blob/main/boxes_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/boxes_light.jpg#gh-light-mode-only)

And each box is measured independently as follows:

The mean pixel intensity in each box, when viewed over time, is a readout for the oscillatory dynamics in that region. For each channel, the period of the oscillatory signal can be estimated by calculating the autocorrelation of that signal.

![GitHub-Mark-Light](https://github.com/zacswider/README_Images/blob/main/autocorrelation_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/autocorrelation_light.jpg#gh-light-mode-only)

For 2-channel datasets, the temporal shift (if any) between the two signals is estimated by calculating the crosscorrelation of the two channels. 

![GitHub-Mark-Light](https://github.com/zacswider/README_Images/blob/main/crosscorrelation_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/crosscorrelation_light.jpg#gh-light-mode-only)


Oscillation properties (e.g., signal peak, signal trough, signal amplitude, temporal duration) can be determined from each waveform. As a precaution for noisy data, which real-world data are more often than not, the signal are smoothed using a Savitzky–Golay filter to avoid quantifying spurious peaks.

![GitHub-Mark-Light](https://github.com/zacswider/README_Images/blob/main/peaks_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/peaks_light.jpg#gh-light-mode-only)

Once each box has been independently quantified, they can be combined to estimate properties of the wave population. For example, in the example above we measured a period of 12 frames, is that measurement representative of the whole sample? By looking at the distribution of all period measurements, we can see that it is. 

![GitHub-Mark-Light](https://github.com/zacswider/README_Images/blob/main/meanACF_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/meanACF_light.jpg#gh-light-mode-only)

Similarly, we can assess the population of signal shift measurements, and oscillation/wave properties.

![GitHub-Mark-Light](https://github.com/zacswider/README_Images/blob/main/meanPeaks_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/meanPeaks_light.jpg#gh-light-mode-only)

Finally, if we choose to compare different groups, we will get file full of plots comparing each signal metric between groups.

![GitHub-Mark-Light](https://github.com/zacswider/README_Images/blob/main/comparisons_dark.jpg#gh-dark-mode-only)![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/comparisons_light.jpg#gh-light-mode-only)


## Running the scripts 

1) If it's not already open, open the terminal or anaconda prompt.
2) If you haven't restarted the terminal since installing your environment, you're already in the correct directory. If you aren't sure, type `pwd` into the terminal and hit enter. Next, type `cd ~/Desktop/signalProcessing-main/2-channel` into the terminal and hit enter to navigate to the unzipped flder.
3) Activate the newly installed environment by typing `conda activate wave_analysis` into the terminal and hitting enter. 
4) Type `python3 signalprocessing2c.py` into the terminal and hit enter to run the script.
5) Next a window will appear asking you for some parameters to adjust:

<img width="496" alt="Screen Shot 2022-01-17 at 8 51 59 AM" src="https://user-images.githubusercontent.com/32859488/149791989-d627fab0-c64d-4690-923b-fb56bbe7cb7f.png">

### Setting parameters:
1) This is the source directory for your analysis. Navigate to it using the "Select source directory button". This directory should have one or more 1 or 2-channel time lapse datasets saved in standard standard `tzcyx` order. 
2) This is the box size used for analysis. Boxes should be large enough to filter out noise, but small enough that they don't over-fill the structures being analyzed and merge multiple wave signals together. A good way to empirically find the apppropriate box size is to open your data in [FIJI](https://imagej.net/software/fiji/), draw a box with the rectangle selection tool, open up the z-axis profile plotter `Image > Stacks > Plot Z-axis Profile`, click the "Live" button, and adjust the box dimensions to find a size that you feel like accurately captures the temporal dynamics. Generally, box size should be roughly equal to spatial wave width. 
3) The is the minimum prominence in the autocorrelation curve to be considered a genuine period. Use the default parameter `0.1` to begin, but if data is noisy, the threshold can be increase to filter out artificial noise peaks. The maximum value for this field is 1. 
4) If you want to compare the population measurements between different groups, enter the groups names in this space. These names *must* be present within the names of the file being processed *and unique* to each group. 
5) If you check this box, a graphical output of the autocorrelation for every box analyzed will be saved to the analysis folder.
6) If you check this box, a graphical output of the crosscorrelation for every box analyzed will be saved to the analysis folder.
7) If you check this box, a graphical output of the wave peak analalysis for every box analyzed will be saved to the analysis folder.
NOTE: Options 6 and 7 and very slow, especially for large images. We recommend only checking these boxes while optimizing the settings to check analysis quality.





