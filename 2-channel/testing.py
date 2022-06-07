from signal_processing_class import SignalProcessor
import numpy as np
import matplotlib.pyplot as plt

im_path = '/Users/bementmbp/Desktop/Scripts/waveAnalysis/2-channel/testDatasets/1_Group1.tif'

sp = SignalProcessor(image_path = im_path, box_size=20)
print(f'num y boxes is {sp.y_boxes}')
print(f'num x boxes is {sp.x_boxes}')
print(f'num channels is {sp.num_channels}')
print(f'num frames is {sp.num_frames}')

p = sp.calc_CCF()
#sp.calc_CCF()
sp.calc_peaks()
frame = sp.summarize_results()
print(list(p.keys()))


'''
means = sp.box_means
print(means[0].shape)
acf_results = sp.calc_ACF(peak_thresh = 0.1)
#print(acf_results)
peak_results = sp.calc_peaks()
for key, values in acf_results.items():
    channel = 1
    if f'Ch{channel}' in key:
        print(key, values[0])

ccf_results = sp.calc_CCF()
for key, values in ccf_results.items():
    print(key, values[0])
'''



