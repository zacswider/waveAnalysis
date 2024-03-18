import housekeeping.housekeeping_functions as hf
from custom_gui import BaseGUI, RollingGUI, KymographGUI

from data_workflows import rolling_workflow
from data_workflows.combined_workflow import combined_workflow

####################################################################################################################################
####################################################################################################################################

# TODO: there is something wrong with importing all waveanalysis modules. will not work if they are like waveanalysis.etc when
# running from the command line. 

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
    plot_indv_ACFs = gui.plot_indv_ACFs
    plot_indv_CCFs = gui.plot_indv_CCFs
    plot_indv_peaks = gui.plot_indv_peaks
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
        plot_indv_CCFs = gui.plot_indv_CCFs
        plot_indv_peaks = gui.plot_indv_peaks
        plot_indv_ACFs = gui.plot_indv_ACFs
        line_width = gui.line_width
        box_shift = gui.box_shift
        group_names = gui.group_names
        acf_peak_thresh = gui.acf_peak_thresh
        # set the analysis type
        analysis_type = "kymograph"

    #make dictionary of parameters for log file use
    log_params = {  "Box Size(px)" : box_size,
                    "Box Shift(px)" : box_shift,
                    "Base Directory" : folder_path,
                    "ACF Peak Prominence" : acf_peak_thresh,
                    "Group Names" : group_names,
                    "Plot Summary ACFs" : plot_summary_ACFs,
                    "Plot Summary CCFs" : plot_summary_CCFs,
                    "Plot Summary Peaks" : plot_summary_peaks,
                    "Plot Individual ACFs" : plot_indv_ACFs,
                    "Plot Individual CCFs" : plot_indv_CCFs,
                    "Plot Individual Peaks" : plot_indv_peaks,
                    "Files Processed" : [],
                    "Files Not Processed" : [],
                    "Errors" : []
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
                        'Submovies Used' : [],
                        "Errors" : []
                } 
    if analysis_type == 'kymograph':
        log_params = {  "Base Directory": folder_path,
                        "Plot Summary ACFs": plot_summary_ACFs,
                        "Plot Summary CCFs": plot_summary_CCFs,
                        "Plot Summary Peaks": plot_summary_peaks,
                        "Plot Individual ACFs": plot_indv_ACFs,
                        "Plot Individual CCFs": plot_indv_CCFs,
                        "Plot Individual Peaks": plot_indv_peaks,  
                        "Line width": line_width,
                        "Group Names" : group_names,
                        "Files Processed": [],
                        "Files Not Processed": [],
                        "Errors" : []
                }
        
    # identify and report errors in GUI input
    log_params = hf.threshold_check(acf_peak_thresh, log_params)
    
    if len(gui.folder_path) < 1 :        
        log_params["Errors"].append("You didn't enter a directory to analyze")        
        
    if analysis_type == "standard":                         
        result_df = combined_workflow(
            folder_path=folder_path,
            group_names=group_names,
            log_params=log_params,
            analysis_type=analysis_type,
            box_size=box_size,
            box_shift=box_shift,
            acf_peak_thresh=acf_peak_thresh,
            plot_summary_ACFs=plot_summary_ACFs,
            plot_summary_CCFs=plot_summary_CCFs,
            plot_summary_peaks=plot_summary_peaks,
            plot_indv_ACFs=plot_indv_ACFs,
            plot_indv_CCFs=plot_indv_CCFs,
            plot_indv_peaks=plot_indv_peaks
        )
        
    if analysis_type == "kymograph":                         
        result_df = combined_workflow(
            folder_path=folder_path,
            group_names=group_names,
            log_params=log_params,
            analysis_type=analysis_type,
            box_size=box_size,
            line_width=box_shift,
            acf_peak_thresh=acf_peak_thresh,
            plot_summary_ACFs=plot_summary_ACFs,
            plot_summary_CCFs=plot_summary_CCFs,
            plot_summary_peaks=plot_summary_peaks,
            plot_indv_ACFs=plot_indv_ACFs,
            plot_indv_CCFs=plot_indv_CCFs,
            plot_indv_peaks=plot_indv_peaks
        )
    
    if analysis_type == "rolling":
        result_df = rolling_workflow(
            folder_path=folder_path,
            log_params=log_params,
            box_size=box_size,
            box_shift=box_shift,
            roll_size=subframe_size,    
            roll_by=subframe_roll,
            acf_peak_thresh=acf_peak_thresh,
        )
    
    result_df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()

print('Done with script!')