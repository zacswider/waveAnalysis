
from signal_processing_class import SignalProcessor
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from tifffile import imread

im_path = '/Users/bementmbp/Desktop/Scripts/waveAnalysis/2-channel/testDatasets/1_Group1.tif'
im_path = '/Users/bementmbp/Desktop/minimal.tif'

# method 2
def calc_CCF(signal_1, signal_2):
    num_frames = signal_1.shape[0]
    cc_curve = np.correlate(signal_1 - signal_1.mean(), signal_2 - signal_2.mean(), mode='full')
    # normalize the curve
    cc_curve = cc_curve / (num_frames * signal_1.std() * signal_2.std())
    # find the peak closes to zero
    peaks, _ = sig.find_peaks(cc_curve)
    peaks_abs = abs(peaks - cc_curve.shape[0]//2)
    delay_index = peaks[np.argmin(peaks_abs)]
    shift = delay_index - cc_curve.shape[0]//2
    return cc_curve, shift

def findBoxMeans(imageArray, boxSize):              #accepts an image array as a parameter, as well as the desired box size
    depth = imageArray.shape[0]                     #number of frames
    yDims = imageArray.shape[1]                     #number of pixels on y-axis
    xDims = imageArray.shape[2]                     #number of pixels on x-axis
    yBoxes = yDims // boxSize                       #returns int result of floor division; number of boxes on the y axis
    xBoxes = xDims // boxSize                       #returns int result of floor division; number of boxes on the x axis
    growingArray = np.zeros((xBoxes, yBoxes, depth))#makes a starting array of 64 bit zeros that can be modified later. shape = (num x boxes, num y boxes, depth of imageStack)

    for x in range(xBoxes):                         #iterates through the number of boxes on the x-axis
        for y in range (yBoxes):                    #iterates through the number of boxes on the y-axis
            boxMean = np.array([np.mean(imageArray[:,(y*boxSize):(y*boxSize+boxSize),(x*boxSize):(x*boxSize+boxSize)], (1,2))])  
            #creates a 2d array of shape (depth, 1) containing the mean values of the px within the box for each slices
            growingArray[x][y] = boxMean            #reassigns zero values at this position to the box mean values
    #growingArray = np.reshape(growingArray, (growingArray.shape[0]*growingArray.shape[1], growingArray.shape[2])) #reshapes to a 2d instead of 3d array
    return(growingArray)                            #returns ndarray of shape (number of boxes, number of frames)

sp = SignalProcessor(image_path = im_path, box_size=20)
print(f'num y boxes is {sp.y_boxes}')
print(f'num x boxes is {sp.x_boxes}')
print(f'num channels is {sp.num_channels}')
print(f'num frames is {sp.num_frames}')

print(f'box means shape is {sp.box_means.shape}')
signal1 = sp.box_means[0,0,:]
signal2 = sp.box_means[0,1,:]

line, num = calc_CCF(signal1, signal2)
print(f'shift is {num}')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(signal1)
ax2.plot(signal2)
ax3.imshow(sp.image[0,0,0], cmap = 'gray')
ax4.imshow(sp.image[0,0,1], cmap = 'gray')
plt.show()




