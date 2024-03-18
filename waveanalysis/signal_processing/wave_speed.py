import napari
import numpy as np

def define_wave_tracks(file_path):
    # asking the user to identify the ring of interest
    filename = file_path.split('/')[-1]
    viewer = napari.Viewer(title=f'Trace a polygon for {filename}. Press "s" to save and close the window')
    viewer.open(file_path)

    shapes_layer = viewer.add_shapes(shape_type='line')#time saving to automatically create the shapes layer

    @viewer.bind_key('s')
    def save_and_close_ROIs(viewer):
        last_shape = viewer.layers['Shapes'].data[0]

        rois = []
        for shape in viewer.layers['Shapes'].data[1:]:
            rois.append(shape)

        viewer.window.close()

        return last_shape, rois
    
    napari.run()

    return save_and_close_ROIs(viewer) #return the polygon vertices coordinates

def calc_wave_speeds(wave_tracks: np.array, pixel_size: float, frame_interval: float):
    wave_speeds = []
    for wave_track in wave_tracks:
        # wave_speed = np.linalg.norm(wave_track[-1] - wave_track[0]) / (len(wave_track) * frame_interval)
        x1, x2 = wave_track[0][1], wave_track[1][1]
        y1, y2 = wave_track[0][0], wave_track[1][0]
        wave_speed = abs(y2-y1 / x2-x1)
        wave_speed = wave_speed * pixel_size[0] / frame_interval
        wave_speeds.append(wave_speed)

    return wave_speeds