import numpy as np
from tifffile import imread, imwrite, TiffFile

class SignalProcessor:
    
    def __init__(self, image_path, box_size):
        self.image_path = image_path
        self.box_size = box_size
        self.image = imread(self.image_path)

        # standardize image dimensions
        with TiffFile(self.image_path) as tif_file:
            metadata = tif_file.imagej_metadata
        self.num_channels = metadata.get('channels', 1)
        self.num_slices = metadata.get('slices', 1)
        self.num_frames = metadata.get('frames', 1)
        print(f'Image dimensions before reshaping {self.image.shape}')
        print(f'number of channels is {self.num_channels}')
        print(f'number of slices is {self.num_slices}')
        print(f'number of frames is {self.num_frames}')
        self.image = self.image.reshape(self.num_frames, 
                                        self.num_slices, 
                                        self.num_channels, 
                                        self.image.shape[-2], 
                                        self.image.shape[-1])
        print(f'Image dimensions after reshaping {self.image.shape}')
        # max project image stack if num_slices > 1
        if self.num_slices > 1:
            self.image = np.max(self.image, axis = 1)
            self.num_slices = 1
            self.image = self.image.reshape(self.num_frames, 
                                        self.num_slices, 
                                        self.num_channels, 
                                        self.image.shape[-2], 
                                        self.image.shape[-1])
            print(f'Image dimensions after projecting {self.image.shape}')

        # calculate number of boxes in each dimension
        self.x_dim = self.image.shape[-1]
        self.y_dim = self.image.shape[-2]
        self.x_boxes = self.x_dim // self.box_size
        self.y_boxes = self.y_dim // self.box_size

        # define the time-axis means for each channel
        self.box_means = np.zeros((self.x_boxes, self.y_boxes, self.num_channels, self.num_frames))
        for channel in range(self.num_channels):
            for x in range(self.x_boxes):
                for y in range(self.y_boxes):
                    self.box_means[x, y, channel] = np.mean(self.image[:, 0, channel, x*self.box_size:(x+1)*self.box_size, y*self.box_size:(y+1)*self.box_size], axis=(1,2))
        # reshape into 2D array
        self.box_means = self.box_means.reshape((self.num_channels, self.x_boxes*self.y_boxes, self.num_frames))




