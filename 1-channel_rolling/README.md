signalProcessing1C_rolling.py is written to batch analyze oscillatory dynamics _over time_ in 1-channel time lapse datasets. The primary dependencies are numpy, pandas, seaborn, scikit-image, scipy, and matplotlib. See the environment.yml or requirements.txt file to create your own environment. This workflow was used to analyze the data for figures 1B', 2B', 3B', 4A', 4E', 5C, 5D, Supp 1C, Supp 4, and Supp 5 in the publication [_Cell cycle and developmental control of cortical excitability in Xenopus laevis_](https://www.biorxiv.org/content/10.1101/2022.02.11.480124v1 "Link to paper on BioRxiv") . Please see [the README for the single time point workflow](https://github.com/zacswider/waveAnalysis/tree/main/2-channel "Link to README") for a detailed description of how each time point is analyzed. 

## Expected Output:

An example ouput can be seen below:

![GitHub-Mark-Dark](https://github.com/zacswider/README_Images/blob/main/rollingOutput_light.jpg)

In brief, the kymograph illustrates the excitable (in one dimension, oscillatory) dynamics found in the developing _X. laevis_ cell cortex. The plot below shows the raw ouput from this script, illustrating the changes in wave period and amplitude over time. 

## Running the script:

The following instructions assume that you just installed Python and made an environment for wave analysis based on [the previous README](https://github.com/zacswider/waveAnalysis "Main README") .

1) If it's not already open, open the terminal or anaconda prompt.
2) If you haven't restarted the terminal since installing your environment, you're already in the correct directory. If you aren't sure, type `cd` into the terminal and hit enter to reset your path. Next, type `cd ~/Desktop/signalProcessing-main/1-channel_rolling` into the terminal and hit enter to navigate to the unzipped flder.
3) Activate the newly installed environment by typing `conda activate wave_analysis` into the terminal and hitting enter. 
4) Type `python3 signalProcessing1C_rolling.py` into the terminal and hit enter to run the script.
5) Next a window will appear asking you for some parameters to adjust:

<img width="496" alt="Example of rolling analysis GUI" src="https://github.com/zacswider/README_Images/blob/main/GUI_Rolling.png">

### Setting parameters:
1) This is the source directory for your analysis. Navigate to it using the "Select source directory button". This directory should have one or more 1 or 2-channel time lapse datasets saved in standard standard `tzcyx` order. 
2) This is the box size used for analysis. Boxes should be large enough to filter out noise, but small enough that they don't over-fill the structures being analyzed and merge multiple wave signals together. A good way to empirically find the apppropriate box size is to open your data in [FIJI](https://imagej.net/software/fiji/), draw a box with the rectangle selection tool, open up the z-axis profile plotter `Image > Stacks > Plot Z-axis Profile`, click the "Live" button, and adjust the box dimensions to find a size that you feel like accurately captures the temporal dynamics. Generally, box size should be roughly equal to spatial wave width. 
3) This is the number of consecutive frames to analyze wave period and peak properties.
4) This is the number of frames to slide the analysis window by.
5) The is the minimum prominence in the autocorrelation curve to be considered a genuine period. Use the default parameter `0.1` to begin, but if data is noisy, the threshold can be increase to filter out artificial noise peaks. The maximum value for this field is 1. 
6) If you check this box, a graphical output of the autocorrelation for every box analyzed will be saved to the analysis folder.
7) If you check this box, a graphical output of the autocorrelation for every window analyzed will be saved to the analysis folder.
8) If you check this box, a graphical output of the wave peak analalysis for every box analyzed will be saved to the analysis folder.
NOTE: Options 6, 7, and 8 are very slow, especially for large images. We recommend only checking these boxes while optimizing the settings to check analysis quality.
