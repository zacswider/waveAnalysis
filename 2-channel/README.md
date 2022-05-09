signalProcessing2C.py is written to batch analyze oscillatory dynamics in both 1-channel and 2-channel time lapse datasets. The primary dependencies are numpy, pandas, seaborn, scikit-image, scipy, and matplotlib. See the environment.yml or requirements.txt file to create your own environment. 

## Running the scripts 

1) If it's not already open, open the terminal or anaconda prompt.
2) If you haven't restarted the terminal since installing your environment, you're already in the correct directory. If you aren't sure, type `cd` into the terminal and hit enter to reset your path. Next, type `cd ~/Desktop/signalProcessing-main/2-channel` into the terminal and hit enter to navigate to the unzipped flder.
3) Activate the newly installed environment by typing `conda activate wave_analysis` into the terminal and hitting enter. 
4) Type `python3 signalprocessing2c.py` into the terminal and hit enter to run the script.
5) Next a window will appear asking you for some parameters to adjust:

<img width="496" alt="Example GUI" src="https://user-images.githubusercontent.com/32859488/149791989-d627fab0-c64d-4690-923b-fb56bbe7cb7f.png">

### Setting parameters:
1) This is the source directory for your analysis. Navigate to it using the "Select source directory button". This directory should have one or more 1 or 2-channel time lapse datasets saved in standard standard `tzcyx` order. 
2) This is the box size used for analysis. Boxes should be large enough to filter out noise, but small enough that they don't over-fill the structures being analyzed and merge multiple wave signals together. A good way to empirically find the apppropriate box size is to open your data in [FIJI](https://imagej.net/software/fiji/), draw a box with the rectangle selection tool, open up the z-axis profile plotter `Image > Stacks > Plot Z-axis Profile`, click the "Live" button, and adjust the box dimensions to find a size that you feel like accurately captures the temporal dynamics. Generally, box size should be roughly equal to spatial wave width. 
3) The is the minimum prominence in the autocorrelation curve to be considered a genuine period. Use the default parameter `0.1` to begin, but if data is noisy, the threshold can be increase to filter out artificial noise peaks. The maximum value for this field is 1. 
4) If you want to compare the population measurements between different groups, enter the groups names in this space. These names *must* be present within the names of the file being processed *and unique* to each group. 
5) If you check this box, a graphical output of the autocorrelation for every box analyzed will be saved to the analysis folder.
6) If you check this box, a graphical output of the crosscorrelation for every box analyzed will be saved to the analysis folder.
7) If you check this box, a graphical output of the wave peak analalysis for every box analyzed will be saved to the analysis folder.
NOTE: Options 6 and 7 and very slow, especially for large images. We recommend only checking these boxes while optimizing the settings to check analysis quality.





