import housekeeping.housekeeping_functions as hf
from custom_gui import BaseGUI, RollingGUI, KymographGUI
from data_workflows.combined_workflow import combined_workflow
from data_workflows.rolling_workflow import rolling_workflow

def main():
    '''
    Main function to run the wave analysis GUI and analysis workflows.
    '''
    # make GUI object and display the window
    gui = BaseGUI()
    gui.mainloop()

    # get standard GUI parameters
    box_size = gui.box_size
    bin_shift = gui.bin_shift
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
        bin_shift = gui.bin_shift
        group_names = gui.group_names
        acf_peak_thresh = gui.acf_peak_thresh
        calc_wave_speeds = gui.calc_wave_speeds
        # set the analysis type
        analysis_type = "kymograph"
    
    if analysis_type == 'rolling':
        log_params = {  "Box Size(px)" : box_size,
                        "Box Shift(px)" : bin_shift,
                        "Base Directory" : folder_path,
                        "ACF Peak Prominence" : acf_peak_thresh,
                        "Plot sub-movie ACFs" : plot_sf_ACFs,
                        "Plot movie CCFs" : plot_sf_CCFs,
                        "Plot movie Peaks" : plot_sf_peaks,
                        'Files Processed': [],
                        'Files Not Processed': [],
                        'Plotting errors': [],
                        'Submovies Used' : [],
                        'Errors': [],
                        'Frame Interval': [],
                        'Pixel Size': []
                } 
 
    # check if a directory was entered
    if len(gui.folder_path) < 1 :        
        log_params["Errors"].append("You didn't enter a directory to analyze")        
        
    # Run the analysis based on the GUI input
    if analysis_type == "standard":              
        summary_df = combined_workflow(
            main_directory=folder_path,
            group_names=group_names,
            analysis_type=analysis_type,
            acf_peak_thresh=acf_peak_thresh,
            plot_summary_ACFs=plot_summary_ACFs,
            plot_summary_CCFs=plot_summary_CCFs,
            plot_summary_peaks=plot_summary_peaks,
            plot_indv_ACFs=plot_indv_ACFs,
            plot_indv_CCFs=plot_indv_CCFs,
            calc_wave_speeds=False,
            plot_wave_speeds=False,
            plot_indv_peaks=plot_indv_peaks,
            box_size=box_size,
            bin_shift=bin_shift,
            line_width=None
        )
    
    if analysis_type == "rolling":
        summary_df = rolling_workflow(
            main_directory=folder_path,
            log_params=log_params,
            box_size=box_size,
            box_shift=box_shift,
            roll_size=subframe_size,    
            roll_by=subframe_roll,
            acf_peak_thresh=acf_peak_thresh,
        )

    if analysis_type == "kymograph":                         
        summary_df = combined_workflow(
            main_directory=folder_path,
            group_names=group_names,
            analysis_type=analysis_type,
            acf_peak_thresh=acf_peak_thresh,
            plot_summary_ACFs=plot_summary_ACFs,
            plot_summary_CCFs=plot_summary_CCFs,
            plot_summary_peaks=plot_summary_peaks,
            plot_indv_ACFs=plot_indv_ACFs,
            plot_indv_CCFs=plot_indv_CCFs,
            plot_indv_peaks=plot_indv_peaks,
            calc_wave_speeds=calc_wave_speeds,
            plot_wave_speeds=True, # always plot wave speeds for now
            box_size=None,
            bin_shift=bin_shift,
            line_width=line_width,
        )
    
if __name__ == '__main__':
    main()

print('Done with script!')