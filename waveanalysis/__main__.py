import sys 
import numpy as np
import matplotlib.pyplot as plt

from waveanalysismods.customgui import BaseGUI, RollingGUI, KymographGUI
from waveanalysis.data_workflows.standard_kymo_workflow import combined_workflow

np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.max_open_warning'] = 0

####################################################################################################################################
####################################################################################################################################

def main():
    '''** GUI Window and sanity checks'''
    # make GUI object and display the window
    gui = BaseGUI()
    gui.mainloop()

    # get standard GUI parameters
    box_size = gui.box_size
    box_shift = gui.box_shift
    folder_path = gui.folder_path
    group_names = gui.group_names
    acf_peak_thresh = gui.acf_peak_thresh
    plot_summary_ACFs = gui.plot_summary_ACFs
    plot_summary_CCFs = gui.plot_summary_CCFs
    plot_summary_peaks = gui.plot_summary_peaks
    plot_ind_ACFs = gui.plot_ind_ACFs
    plot_ind_CCFs = gui.plot_ind_CCFs
    plot_ind_peaks = gui.plot_ind_peaks

    # set parameters to 'None' unless their specific GUIs are opened
    subframe_size = None
    subframe_roll = None
    line_width = None

    # set the analysis type
    analysis_type = "standard"

    # if rolling GUI specified, make rolling GUI object and display the window
    if gui.rolling:
        # make GUI object and display the window
        gui = RollingGUI()
        gui.mainloop()

        # get GUI parameters
        box_size = gui.box_size
        box_shift = gui.box_shift
        folder_path = gui.folder_path
        acf_peak_thresh = gui.acf_peak_thresh
        plot_sf_ACFs = gui.plot_sf_ACFs
        plot_sf_CCFs = gui.plot_sf_CCFs
        plot_sf_peaks = gui.plot_sf_peaks
        subframe_size = gui.subframe_size
        subframe_roll = gui.subframe_roll

        # set these to None to make sure they are not triggered in the script
        line_width = None
        group_names = ['']

        # set the analysis type
        analysis_type = "rolling"

    # if kymograph GUI specified, make kymograph GUI object and display the window
    if gui.kymograph:
        # make GUI object and display the window
        gui = KymographGUI()
        gui.mainloop()

        # get GUI parameters
        folder_path = gui.folder_path
        plot_summary_CCFs = gui.plot_summary_CCFs
        plot_summary_peaks = gui.plot_summary_peaks
        plot_summary_ACFs = gui.plot_summary_ACFs
        plot_ind_CCFs = gui.plot_ind_CCFs
        plot_ind_peaks = gui.plot_ind_peaks
        plot_ind_ACFs = gui.plot_ind_ACFs
        line_width = gui.line_width
        box_shift = gui.box_shift
        group_names = gui.group_names
        acf_peak_thresh = gui.acf_peak_thresh

        # set these to None to make sure they are not triggered in the script
        subframe_size = None
        subframe_roll = None
        box_size = None

        # set the analysis type
        analysis_type = "kymograph"

    # identify and report errors in GUI input
    errors = []
    if gui.acf_peak_thresh > 1 :                                                            # type: ignore
        errors.append("The ACF peak prominence can not be greater than 1",
                    ", set 'ACF peak prominence threshold' to a value between 0 and 1.",    # type: ignore
                    "More realistically, a value between 0 and 0.5")
    if len(gui.folder_path) < 1 :                                                           # type: ignore
        errors.append("You didn't enter a directory to analyze")

    if len(errors) >= 1 :
        print("Error Log:")
        for count, error in enumerate(errors):
            print(count,":", error)
        sys.exit("Please fix errors and try again.") 

    #make dictionary of parameters for log file use
    log_params = {  "Box Size(px)" : box_size,
                    "Box Shift(px)" : box_shift,
                    "Base Directory" : folder_path,
                    "ACF Peak Prominence" : acf_peak_thresh,
                    "Group Names" : group_names,
                    "Plot Summary ACFs" : plot_summary_ACFs,
                    "Plot Summary CCFs" : plot_summary_CCFs,
                    "Plot Summary Peaks" : plot_summary_peaks,
                    "Plot Individual ACFs" : plot_ind_ACFs,
                    "Plot Individual CCFs" : plot_ind_CCFs,
                    "Plot Individual Peaks" : plot_ind_peaks,
                    "Group Matching Errors" : [],
                    "Files Processed" : [],
                    "Files Not Processed" : [],
                    'Plotting errors' : []
                } 
    if analysis_type == 'rolling':
        log_params = {  "Box Size(px)" : box_size,
                        "Box Shift(px)" : box_shift,
                        "Base Directory" : folder_path,
                        "ACF Peak Prominence" : acf_peak_thresh,
                        "Plot sub-movie ACFs" : plot_sf_ACFs,
                        "Plot movie CCFs" : plot_sf_CCFs,
                        "Plot movie Peaks" : plot_sf_peaks,
                        "Files Processed" : [],
                        "Files Not Processed" : [],
                        'Submovies Used' : []
                } 
    if analysis_type == 'kymograph':
        log_params = {"Base Directory": folder_path,
                  "Plot Summary ACFs": plot_summary_ACFs,
                "Plot Summary CCFs": plot_summary_CCFs,
                "Plot Summary Peaks": plot_summary_peaks,
                "Plot Individual ACFs": plot_ind_ACFs,
                "Plot Individual CCFs": plot_ind_CCFs,
                "Plot Individual Peaks": plot_ind_peaks,  
                "Line width": line_width,
                "Group Names" : group_names,
                "Files Processed": [],
                "Files Not Processed": [],
                'Plotting errors': [],
                "Group Matching Errors" : []
                }

    result_df = combined_workflow(
        str(folder_path),   # TODO find a way to get accurate type hints out of tk gui so my linter doesn't blow up
        group_names,        # type: ignore
        log_params,
        analysis_type,
        box_size,           # type: ignore
        box_shift,          # type: ignore
        subframe_size,      # type: ignore
        subframe_roll,      # type: ignore
        line_width,         # type: ignore
        acf_peak_thresh,    # type: ignore
        plot_summary_ACFs,  # type: ignore
        plot_summary_CCFs,  # type: ignore
        plot_summary_peaks, # type: ignore
        plot_ind_ACFs,      # type: ignore
        plot_ind_CCFs,      # type: ignore
        plot_ind_peaks,     # type: ignore
    )
    result_df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()

print('Done with script!')