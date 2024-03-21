import napari
import numpy as np

def define_wave_tracks(file_path):
    '''
    Defines wave tracks by allowing the user to trace a polygon on an image.

    Parameters:
        file_path (str): The path to the image file.

    Returns:
        list: A list of polygon vertices coordinates representing the wave tracks.
    '''
    # asking the user to identify the ring of interest
    filename = file_path.split('/')[-1]
    viewer = napari.Viewer(title=f'Trace a polygon for {filename}. Press "s" to save and close the window')
    viewer.open(file_path)

    # add a shapes layer to the viewer
    shapes_layer = viewer.add_shapes()

    @viewer.bind_key('s') # user presses "s" to save and close the window
    def save_and_close_ROIs(viewer):
        '''
        Save and close the ROIs.
        '''
        rois = []
        for shape in viewer.layers['Shapes'].data:
            rois.append(shape)
        viewer.window.close()

        return rois
    
    napari.run()

    return save_and_close_ROIs(viewer) #return the polygon vertices coordinates

def calc_wave_speeds(wave_tracks: np.array, pixel_size: float, frame_interval: float):
    '''
    Calculate the wave speeds for a given set of wave tracks.

    Parameters:
    wave_tracks (np.array): An array containing wave tracks.
    pixel_size (float): The size of each pixel in the image.
    frame_interval (float): The time interval between frames.

    Returns:
    wave_speeds (list): A list of wave speeds calculated for each wave track.
    '''
    wave_speeds = []
    for wave_track in wave_tracks:
        # get the x and y coordinates of the wave track
        x1, x2 = wave_track[0][1], wave_track[1][1]
        y1, y2 = wave_track[0][0], wave_track[1][0]
        # calculate the wave speed
        wave_speed = abs(y2 - y1 / x2 - x1)
        wave_speed = wave_speed * pixel_size[0] / frame_interval
        wave_speeds.append(wave_speed)

    return wave_speeds