import sys
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

class CustomGUI(tk.Tk):
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
        self.plot_ind_ACFs = tk.BooleanVar()
        self.plot_ind_CCFs = tk.BooleanVar()
        self.plot_ind_peaks = tk.BooleanVar()
        self.acf_peak_thresh = tk.DoubleVar()
        self.acf_peak_thresh.set(0.1)
        self.group_names = tk.StringVar()
        self.folder_path = tk.StringVar()

        # file path selection widget
        self.file_path_entry = ttk.Entry(self, textvariable = self.folder_path)
        self.file_path_entry.grid(row = 0, column = 0, padx = 10, sticky = 'E')
        self.file_path_button = ttk.Button(self, text = 'Select folder')
        # make a default path
        self.folder_path.set('/Users/bementmbp/Desktop/Scripts/waveAnalysis/2-channel/testDatasets/')
        self.file_path_button['command'] = self.get_folder_path
        self.file_path_button.grid(row = 0, column = 1, padx = 10, sticky = 'W')        

        # box size selection widget
        self.box_size_entry = ttk.Entry(self, width = 3, textvariable = self.box_size)
        self.box_size_entry.grid(row = 1, column = 0, padx = 10, sticky = 'E')
        # create box size label text
        self.box_size_label = ttk.Label(self, text = 'Box size (pixels)')
        self.box_size_label.grid(row = 1, column = 1, padx = 10, sticky = 'W')

        # create ACF peak threshold entry widget
        self.acf_peak_thresh_entry = ttk.Entry(self, width = 3, textvariable = self.acf_peak_thresh)
        self.acf_peak_thresh_entry.grid(row = 2, column = 0, padx = 10, sticky = 'E')
        # create ACF peak threshold label text
        self.acf_peak_thresh_label = ttk.Label(self, text = 'ACF peak threshold')
        self.acf_peak_thresh_label.grid(row = 2, column = 1, padx = 10, sticky = 'W')

        # create group names entry widget
        self.group_names_entry = ttk.Entry(self, textvariable = self.group_names)
        self.group_names_entry.grid(row = 3, column = 0, padx = 10, sticky = 'E')
        # create group names label text
        self.group_names_label = ttk.Label(self, text = 'Group names')
        self.group_names_label.grid(row = 3, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting individual ACFs
        self.plot_ind_ACFs_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_ACFs)
        self.plot_ind_ACFs_checkbox.grid(row = 4, column = 0, padx = 10, sticky = 'E')
        self.plot_ind_ACFs_label = ttk.Label(self, text = 'Plot individual ACFs')
        self.plot_ind_ACFs_label.grid(row = 4, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting individual CCFs
        self.plot_ind_CCFs_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_CCFs)
        self.plot_ind_CCFs_checkbox.grid(row = 5, column = 0, padx = 10, sticky = 'E')
        self.plot_ind_CCFs_label = ttk.Label(self, text = 'Plot individual CCFs')
        self.plot_ind_CCFs_label.grid(row = 5, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting individual peaks
        self.plot_ind_peaks_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_peaks)
        self.plot_ind_peaks_checkbox.grid(row = 6, column = 0, padx = 10, sticky = 'E')
        self.plot_ind_peaks_label = ttk.Label(self, text = 'Plot individual peaks')
        self.plot_ind_peaks_label.grid(row = 6, column = 1, padx = 10, sticky = 'W')

        # create start button
        self.start_button = ttk.Button(self, text = 'Start analysis')
        self.start_button['command'] = self.start_analysis
        self.start_button.grid(row = 7, column = 0, padx = 10, sticky = 'E')

        # create cancel button
        self.cancel_button = ttk.Button(self, text = 'Cancel')
        self.cancel_button['command'] = self.cancel_analysis
        self.cancel_button.grid(row = 7, column = 1, padx = 10, sticky = 'W')

    def get_folder_path(self):
        self.folder_path.set(askdirectory())

    def cancel_analysis(self):
        sys.exit('You have cancelled the analysis')
    
    def start_analysis(self):
        # get the values stored in the widget
        self.box_size = self.box_size.get()
        self.acf_peak_thresh = self.acf_peak_thresh.get()
        self.group_names = self.group_names.get()
        self.plot_ind_ACFs = self.plot_ind_ACFs.get()
        self.plot_ind_CCFs = self.plot_ind_CCFs.get()
        self.plot_ind_peaks = self.plot_ind_peaks.get()
        self.folder_path = self.folder_path.get()
        
        # convert group names to list of strings
        self.group_names = [group_name.strip() for group_name in self.group_names.split(',')]

        # destroy the widget
        self.destroy()