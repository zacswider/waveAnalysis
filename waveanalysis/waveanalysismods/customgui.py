import sys
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

class BaseGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure root window
        self.title("Define your analysis parameters")
        self.geometry("630x265")
        
        #sets number of columns in the main window
        self.columnconfigure(0, weight = 1)
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 1)

        # define variable types for the different widget field
        self.box_size = tk.IntVar()
        self.box_size.set(20)
        self.box_shift = tk.IntVar()
        self.box_shift.set(20)
        self.plot_summary_ACFs = tk.BooleanVar()
        self.plot_summary_ACFs.set(True)
        self.plot_summary_CCFs = tk.BooleanVar()
        self.plot_summary_CCFs.set(True)
        self.plot_summary_peaks = tk.BooleanVar()
        self.plot_summary_peaks.set(True)
        self.plot_ind_ACFs = tk.BooleanVar()
        self.plot_ind_ACFs.set(False)
        self.plot_ind_CCFs = tk.BooleanVar()
        self.plot_ind_CCFs.set(False)
        self.plot_ind_peaks = tk.BooleanVar()
        self.plot_ind_peaks.set(False)
        self.acf_peak_thresh = tk.DoubleVar()
        self.acf_peak_thresh.set(0.1)
        self.group_names = tk.StringVar()
        self.group_names.set('001,002')
        self.folder_path = tk.StringVar()
        # set default value for 'rolling' and 'kymograph' to False
        self.rolling = False
        self.kymograph = False

        # file path selection widget
        self.file_path_entry = ttk.Entry(self, textvariable = self.folder_path)
        self.file_path_entry.grid(row = 0, column = 0, padx = 10, sticky = 'E')
        self.file_path_button = ttk.Button(self, text = 'Select folder')
        # make a default path
        self.folder_path.set('/Users/domchom/Desktop/wave_analysis_testing/en-face')
        self.file_path_button['command'] = self.get_folder_path
        self.file_path_button.grid(row = 0, column = 1, padx = 10, sticky = 'W')        

        # box size selection widget
        self.box_size_entry = ttk.Entry(self, width = 3, textvariable = self.box_size)
        self.box_size_entry.grid(row = 1, column = 0, padx = 10, sticky = 'E')
        # create box size label text
        self.box_size_label = ttk.Label(self, text = 'Box size (pixels)')
        self.box_size_label.grid(row = 1, column = 1, padx = 10, sticky = 'W')

        self.box_shift_entry = ttk.Entry(self, width = 3, textvariable = self.box_shift)
        self.box_shift_entry.grid(row = 2, column = 0, padx = 10, sticky = 'E')
        # create box size label text
        self.box_shift_label = ttk.Label(self, text = 'Box shift (pixels)')
        self.box_shift_label.grid(row = 2, column = 1, padx = 10, sticky = 'W')

        # create ACF peak threshold entry widget
        self.acf_peak_thresh_entry = ttk.Entry(self, width = 3, textvariable = self.acf_peak_thresh)
        self.acf_peak_thresh_entry.grid(row = 3, column = 0, padx = 10, sticky = 'E')
        # create ACF peak threshold label text
        self.acf_peak_thresh_label = ttk.Label(self, text = 'ACF peak threshold')
        self.acf_peak_thresh_label.grid(row = 3, column = 1, padx = 10, sticky = 'W')

        # create group names entry widget
        self.group_names_entry = ttk.Entry(self, textvariable = self.group_names)
        self.group_names_entry.grid(row = 4, column = 0, padx = 10, sticky = 'E')
        # create group names label text
        self.group_names_label = ttk.Label(self, text = 'Group names')
        self.group_names_label.grid(row = 4, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting summary ACFs
        self.plot_summary_ACFs_checkbox = ttk.Checkbutton(self, variable = self.plot_summary_ACFs)
        self.plot_summary_ACFs_checkbox.grid(row = 5, column = 0, padx = 10, sticky = 'E')
        self.plot_summary_ACFs_label = ttk.Label(self, text = 'Plot summary ACFs')
        self.plot_summary_ACFs_label.grid(row = 5, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting summary CCFs
        self.plot_summary_CCFs_checkbox = ttk.Checkbutton(self, variable = self.plot_summary_CCFs)
        self.plot_summary_CCFs_checkbox.grid(row = 6, column = 0, padx = 10, sticky = 'E')
        self.plot_summary_CCFs_label = ttk.Label(self, text = 'Plot summary CCFs')
        self.plot_summary_CCFs_label.grid(row = 6, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting summary peaks
        self.plot_summary_peaks_checkbox = ttk.Checkbutton(self, variable = self.plot_summary_peaks)
        self.plot_summary_peaks_checkbox.grid(row = 7, column = 0, padx = 10, sticky = 'E')
        self.plot_summary_peaks_label = ttk.Label(self, text = 'Plot summary peaks')
        self.plot_summary_peaks_label.grid(row = 7, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting individual ACFs
        self.plot_ind_ACFs_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_ACFs)
        self.plot_ind_ACFs_checkbox.grid(row = 5, column = 2, padx = 10, sticky = 'E')
        self.plot_ind_ACFs_label = ttk.Label(self, text = 'Plot individual ACFs')
        self.plot_ind_ACFs_label.grid(row = 5, column = 3, padx = 10, sticky = 'W')

        # create checkbox for plotting individual CCFs
        self.plot_ind_CCFs_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_CCFs)
        self.plot_ind_CCFs_checkbox.grid(row = 6, column = 2, padx = 10, sticky = 'E')
        self.plot_ind_CCFs_label = ttk.Label(self, text = 'Plot individual CCFs')
        self.plot_ind_CCFs_label.grid(row = 6, column = 3, padx = 10, sticky = 'W')

        # create checkbox for plotting individual peaks
        self.plot_ind_peaks_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_peaks)
        self.plot_ind_peaks_checkbox.grid(row = 7, column = 2, padx = 10, sticky = 'E')
        self.plot_ind_peaks_label = ttk.Label(self, text = 'Plot individual peaks')
        self.plot_ind_peaks_label.grid(row = 7, column = 3, padx = 10, sticky = 'W')
        
        # create start button
        self.start_button = ttk.Button(self, text = 'Start analysis')
        self.start_button['command'] = self.start_analysis
        self.start_button.grid(row = 9, column = 0, padx = 10, sticky = 'E')

        # create cancel button
        self.cancel_button = ttk.Button(self, text = 'Cancel')
        self.cancel_button['command'] = self.cancel_analysis
        self.cancel_button.grid(row = 9, column = 1, padx = 10, sticky = 'W')

        # create button to launch rolling analysis gui
        self.rolling_button = ttk.Button(self, text = 'Launch rolling analysis')
        self.rolling_button['command'] = self.launch_rolling_analysis
        self.rolling_button.grid(row = 9, column = 3, padx = 10, sticky = 'E')

        # create button to launch kymograph analysis gui
        self.kymograph_button = ttk.Button(self, text = 'Launch kymograph analysis')
        self.kymograph_button['command'] = self.launch_kymograph_analysis
        self.kymograph_button.grid(row = 10, column = 3, padx = 10, sticky = 'E')

    def get_folder_path(self):
        self.folder_path.set(askdirectory())

    def launch_rolling_analysis(self):
        self.rolling = True
        self.kymograph = False
        self.destroy()

    def launch_kymograph_analysis(self):
        self.kymograph = True
        self.rolling = False
        self.destroy()

    def cancel_analysis(self):
        sys.exit('You have cancelled the analysis')
    
    def start_analysis(self):
        # get the values stored in the widget
        self.box_size = self.box_size.get()
        self.box_shift = self.box_shift.get()
        self.acf_peak_thresh = self.acf_peak_thresh.get()
        self.group_names = self.group_names.get()
        self.plot_summary_ACFs = self.plot_summary_ACFs.get()
        self.plot_summary_CCFs = self.plot_summary_CCFs.get()
        self.plot_summary_peaks = self.plot_summary_peaks.get()
        self.plot_ind_ACFs = self.plot_ind_ACFs.get()
        self.plot_ind_CCFs = self.plot_ind_CCFs.get()
        self.plot_ind_peaks = self.plot_ind_peaks.get()
        self.folder_path = self.folder_path.get()
        
        # convert group names to list of strings
        self.group_names = [group_name.strip() for group_name in self.group_names.split(',')]

        # destroy the widget
        self.destroy()

class RollingGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure root window
        self.title("Define your analysis parameters")
        self.geometry("500x250")

        #sets number of columns in the main window
        self.columnconfigure(0, weight = 1)
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 1)

        # define variable types for the different widget field
        self.box_size = tk.IntVar()
        self.box_size.set(20)
        self.box_shift = tk.IntVar()
        self.box_shift.set(20)
        self.subframe_size = tk.IntVar()
        self.subframe_size.set(50)
        self.subframe_roll = tk.IntVar()
        self.subframe_roll.set(5)
        self.plot_sf_ACFs = tk.BooleanVar()
        self.plot_sf_CCFs = tk.BooleanVar()
        self.plot_sf_peaks = tk.BooleanVar()
        self.acf_peak_thresh = tk.DoubleVar()
        self.acf_peak_thresh.set(0.1)
        self.folder_path = tk.StringVar()

        # file path selection widget
        self.file_path_entry = ttk.Entry(self, textvariable = self.folder_path)
        self.file_path_entry.grid(row = 0, column = 0, padx = 10, sticky = 'E')
        self.file_path_button = ttk.Button(self, text = 'Select folder')
        # make a default path
        self.folder_path.set('/Users/domchom/Desktop/rolling')
        self.file_path_button['command'] = self.get_folder_path
        self.file_path_button.grid(row = 0, column = 1, padx = 10, sticky = 'W')        

        # box size selection widget
        self.box_size_entry = ttk.Entry(self, width = 3, textvariable = self.box_size)
        self.box_size_entry.grid(row = 1, column = 0, padx = 10, sticky = 'E')
        # create box size label text
        self.box_size_label = ttk.Label(self, text = 'Box size (pixels)')
        self.box_size_label.grid(row = 1, column = 1, padx = 10, sticky = 'W')

        # box shift selection widget
        self.box_shift_entry = ttk.Entry(self, width = 3, textvariable = self.box_shift)
        self.box_shift_entry.grid(row = 2, column = 0, padx = 10, sticky = 'E')
        # create box shift label text
        self.box_shift_label = ttk.Label(self, text = 'Box shift (pixels)')
        self.box_shift_label.grid(row = 2, column = 1, padx = 10, sticky = 'W')

        # subframe size selection widget
        self.subframe_size_entry = ttk.Entry(self, width = 3, textvariable = self.subframe_size)
        self.subframe_size_entry.grid(row = 3, column = 0, padx = 10, sticky = 'E')
        # create subframe size label text
        self.subframe_size_label = ttk.Label(self, text = 'Num frames per sub-movie')
        self.subframe_size_label.grid(row = 3, column = 1, padx = 10, sticky = 'W')

        # subframe roll selection widget
        self.subframe_roll_entry = ttk.Entry(self, width = 3, textvariable = self.subframe_roll)
        self.subframe_roll_entry.grid(row = 4, column = 0, padx = 10, sticky = 'E')
        # create subframe roll label text
        self.subframe_roll_label = ttk.Label(self, text = 'Num frames to roll by')
        self.subframe_roll_label.grid(row = 4, column = 1, padx = 10, sticky = 'W')

        # create ACF peak threshold entry widget
        self.acf_peak_thresh_entry = ttk.Entry(self, width = 3, textvariable = self.acf_peak_thresh)
        self.acf_peak_thresh_entry.grid(row = 5, column = 0, padx = 10, sticky = 'E')
        # create ACF peak threshold label text
        self.acf_peak_thresh_label = ttk.Label(self, text = 'ACF peak threshold')
        self.acf_peak_thresh_label.grid(row = 5, column = 1, padx = 10, sticky = 'W')
        ''' # making this mandatory for the moment
        # create checkbox for plotting subframe ACFs
        self.plot_sf_ACFs_checkbox = ttk.Checkbutton(self, variable = self.plot_sf_ACFs)
        self.plot_sf_ACFs_checkbox.grid(row = 5, column = 0, padx = 10, sticky = 'E')
        self.plot_sf_ACFs_label = ttk.Label(self, text = 'Plot sub-movie ACFs')
        self.plot_sf_ACFs_label.grid(row = 5, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting subframe CCFs
        self.plot_sf_CCFs_checkbox = ttk.Checkbutton(self, variable = self.plot_sf_CCFs)
        self.plot_sf_CCFs_checkbox.grid(row = 6, column = 0, padx = 10, sticky = 'E')
        self.plot_sf_CCFs_label = ttk.Label(self, text = 'Plot sub-movie CCFs')
        self.plot_sf_CCFs_label.grid(row = 6, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting subframe peaks
        self.plot_sf_peaks_checkbox = ttk.Checkbutton(self, variable = self.plot_sf_peaks)
        self.plot_sf_peaks_checkbox.grid(row = 7, column = 0, padx = 10, sticky = 'E')
        self.plot_sf_peaks_label = ttk.Label(self, text = 'Plot sub-movie peaks')
        self.plot_sf_peaks_label.grid(row = 7, column = 1, padx = 10, sticky = 'W')
        '''
        # create start button
        self.start_button = ttk.Button(self, text = 'Start analysis')
        self.start_button['command'] = self.start_analysis
        self.start_button.grid(row = 8, column = 0, padx = 10, sticky = 'E')

        # create cancel button
        self.cancel_button = ttk.Button(self, text = 'Cancel')
        self.cancel_button['command'] = self.cancel_analysis
        self.cancel_button.grid(row = 8, column = 1, padx = 10, sticky = 'W')

    def get_folder_path(self):
        self.folder_path.set(askdirectory())

    def cancel_analysis(self):
        sys.exit('You have cancelled the analysis')
    
    def start_analysis(self):
        # get the values stored in the widget
        self.box_size = self.box_size.get()
        self.box_shift = self.box_shift.get()
        self.acf_peak_thresh = self.acf_peak_thresh.get()
        self.plot_sf_ACFs = self.plot_sf_ACFs.get()
        self.plot_sf_CCFs = self.plot_sf_CCFs.get()
        self.plot_sf_peaks = self.plot_sf_peaks.get()
        self.folder_path = self.folder_path.get()
        self.subframe_size = self.subframe_size.get()
        self.subframe_roll = self.subframe_roll.get()

        self.kymograph = False

        # destroy the widget
        self.destroy()

class KymographGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure root window
        self.title("Define your analysis parameters")
        self.geometry("600x200")
        
        #sets number of columns in the main window
        self.columnconfigure(0, weight = 1)
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 1)

        # define variable types for the different widget field
        self.line_width = tk.IntVar()
        self.line_width.set(3)
        self.plot_summary_ACFs = tk.BooleanVar()
        self.plot_summary_ACFs.set(True)
        self.plot_summary_CCFs = tk.BooleanVar()
        self.plot_summary_CCFs.set(True)
        self.plot_summary_peaks = tk.BooleanVar()
        self.plot_summary_peaks.set(True)
        self.acf_peak_thresh = tk.DoubleVar()
        self.acf_peak_thresh.set(0.1)

        self.plot_ind_ACFs = tk.BooleanVar()
        self.plot_ind_ACFs.set(False)
        self.plot_ind_CCFs = tk.BooleanVar()
        self.plot_ind_CCFs.set(False)
        self.plot_ind_peaks = tk.BooleanVar()
        self.plot_ind_peaks.set(False)
        self.group_names = tk.StringVar()
        self.group_names.set("003,007")
        self.folder_path = tk.StringVar()

        # file path selection widget
        self.file_path_entry = ttk.Entry(self, textvariable = self.folder_path)
        self.file_path_entry.grid(row = 0, column = 0, padx = 10, sticky = 'E')
        self.file_path_button = ttk.Button(self, text = 'Select folder')

        # make a default path
        self.folder_path.set('/Users/domchom/Desktop/wave_analysis_testing/kymo')
        self.file_path_button['command'] = self.get_folder_path
        self.file_path_button.grid(row = 0, column = 1, padx = 10, sticky = 'W')        

        # box size selection widget
        self.line_width_entry = ttk.Entry(self, width = 3, textvariable = self.line_width)
        self.line_width_entry.grid(row = 1, column = 0, padx = 10, sticky = 'E')
        # create box size label text
        self.line_width_label = ttk.Label(self, text = 'Line width (pixels)')
        self.line_width_label.grid(row = 1, column = 1, padx = 10, sticky = 'W')

        # create group names entry widget
        self.group_names_entry = ttk.Entry(self, textvariable = self.group_names)
        self.group_names_entry.grid(row = 4, column = 0, padx = 10, sticky = 'E')
        # create group names label text
        self.group_names_label = ttk.Label(self, text = 'Group names')
        self.group_names_label.grid(row = 4, column = 1, padx = 10, sticky = 'W')

        # create ACF peak threshold entry widget
        self.acf_peak_thresh_entry = ttk.Entry(self, width = 3, textvariable = self.acf_peak_thresh)
        self.acf_peak_thresh_entry.grid(row = 5, column = 0, padx = 10, sticky = 'E')
        self.acf_peak_thresh_label = ttk.Label(self, text = 'ACF peak threshold')
        self.acf_peak_thresh_label.grid(row = 5, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting summary ACFs
        self.plot_summary_ACFs_checkbox = ttk.Checkbutton(self, variable = self.plot_summary_ACFs)
        self.plot_summary_ACFs_checkbox.grid(row = 6, column = 0, padx = 10, sticky = 'E')
        self.plot_summary_ACFs_label = ttk.Label(self, text = 'Plot summary ACFs')
        self.plot_summary_ACFs_label.grid(row = 6, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting summary CCFs
        self.plot_summary_CCFs_checkbox = ttk.Checkbutton(self, variable = self.plot_summary_CCFs)
        self.plot_summary_CCFs_checkbox.grid(row = 7, column = 0, padx = 10, sticky = 'E')
        self.plot_summary_CCFs_label = ttk.Label(self, text = 'Plot summary CCFs')
        self.plot_summary_CCFs_label.grid(row = 7, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting summary peaks
        self.plot_summary_peaks_checkbox = ttk.Checkbutton(self, variable = self.plot_summary_peaks)
        self.plot_summary_peaks_checkbox.grid(row = 8, column = 0, padx = 10, sticky = 'E')
        self.plot_summary_peaks_label = ttk.Label(self, text = 'Plot summary peaks')
        self.plot_summary_peaks_label.grid(row = 8, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting individual ACFs
        self.plot_ind_ACFs_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_ACFs)
        self.plot_ind_ACFs_checkbox.grid(row = 6, column = 2, padx = 10, sticky = 'E')
        self.plot_ind_ACFs_label = ttk.Label(self, text = 'Plot individual ACFs')
        self.plot_ind_ACFs_label.grid(row = 6, column = 3, padx = 10, sticky = 'W')

        # create checkbox for plotting individual CCFs
        self.plot_ind_CCFs_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_CCFs)
        self.plot_ind_CCFs_checkbox.grid(row = 7, column = 2, padx = 10, sticky = 'E')
        self.plot_ind_CCFs_label = ttk.Label(self, text = 'Plot individual CCFs')
        self.plot_ind_CCFs_label.grid(row = 7, column = 3, padx = 10, sticky = 'W')

        # create checkbox for plotting individual peaks
        self.plot_ind_peaks_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_peaks)
        self.plot_ind_peaks_checkbox.grid(row = 8, column = 2, padx = 10, sticky = 'E')
        self.plot_ind_peaks_label = ttk.Label(self, text = 'Plot individual peaks')
        self.plot_ind_peaks_label.grid(row = 8, column = 3, padx = 10, sticky = 'W')
        
        # create start button
        self.start_button = ttk.Button(self, text = 'Start analysis')
        self.start_button['command'] = self.start_analysis
        self.start_button.grid(row = 9, column = 0, padx = 10, sticky = 'E')

        # create cancel button
        self.cancel_button = ttk.Button(self, text = 'Cancel')
        self.cancel_button['command'] = self.cancel_analysis
        self.cancel_button.grid(row = 9, column = 1, padx = 10, sticky = 'W')

    def get_folder_path(self):
        self.folder_path.set(askdirectory())

    def cancel_analysis(self):
        sys.exit('You have cancelled the analysis')
    
    def start_analysis(self):
        # get the values stored in the widget
        self.line_width = self.line_width.get()
        self.group_names = self.group_names.get()
        self.plot_summary_ACFs = self.plot_summary_ACFs.get()
        self.plot_summary_CCFs = self.plot_summary_CCFs.get()
        self.plot_summary_peaks = self.plot_summary_peaks.get()
        self.plot_ind_ACFs = self.plot_ind_ACFs.get()
        self.plot_ind_CCFs = self.plot_ind_CCFs.get()
        self.plot_ind_peaks = self.plot_ind_peaks.get()
        self.folder_path = self.folder_path.get()
        self.acf_peak_thresh = self.acf_peak_thresh.get()

        self.rolling = False
        
        # convert group names to list of strings
        self.group_names = [group_name.strip() for group_name in self.group_names.split(',')]

        # destroy the widget
        self.destroy()