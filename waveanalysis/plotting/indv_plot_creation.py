import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from waveanalysis.signal_processing import normalize_signal

def plot_indv_peak_workflow(
	bin_values: np.ndarray,
	img_prop_dict: dict,
	indv_peak_props: dict,
	num_frames: int
) -> dict:
	
	num_channels = img_prop_dict['num_channels']
	num_bins = img_prop_dict['num_bins']
	analysis_type = img_prop_dict['analysis_type']
	frame_interval = img_prop_dict['frame_interval']

	indv_peak_figs = {}
	
	its = num_channels*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind peaks')
		for channel in range(num_channels):
			for bin in range(num_bins):
				pbar.update(1)
				to_plot = bin_values[:,channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
				# Generate and store the figure for the current channel and bin
				indv_peak_figs[f'Ch{channel + 1} Bin {bin + 1} Peak Props'] = return_indv_peak_prop_figure(
					bin_signal=to_plot,
					prop_dict=indv_peak_props[f'Ch {channel} Bin {bin}'],
					Ch_name=f'Ch{channel + 1} Bin {bin + 1}',
					frame_interval=frame_interval,
					num_frames=num_frames
					)
	
	return indv_peak_figs

def return_indv_peak_prop_figure(
	bin_signal: np.ndarray, 
	prop_dict: dict, 
	Ch_name: str,
	frame_interval: float,
	num_frames: int
) -> plt.Figure:

# Extract peak properties from the dictionary
	smoothed_signal = prop_dict['smoothed']
	peaks = prop_dict['peaks']
	proms = prop_dict['proms']
	heights = prop_dict['heights']
	leftIndex = prop_dict['leftIndex']
	rightIndex = prop_dict['rightIndex']
	midpoints = prop_dict['midpoints']

	# Create the figure and plot raw and smoothed signals
	fig, ax = plt.subplots()
	x_axis = np.arange(0, num_frames) * frame_interval
	ax.plot(x_axis, bin_signal, color = 'tab:gray', label = 'raw signal')
	ax.plot(x_axis, smoothed_signal, color = 'tab:cyan', label = 'smoothed signal')

	# Plot each peak width and amplitude
	if not np.isnan(peaks).any():
		for i in range(peaks.shape[0]):
			ax.hlines(heights[i], 
					leftIndex[i], 
					rightIndex[i], 
					color='tab:olive', 
					linestyle = '-')
			ax.vlines(peaks[i], 
					smoothed_signal[peaks[i]]-proms[i],
					smoothed_signal[peaks[i]], 
					color='tab:purple', 
					linestyle = '-')

			ax.hlines(heights[i]-5, 
					peaks[i], 
					midpoints[i], 
					color='tab:orange', 
					linestyle = '-')

		# Plot the legend for the first peak
		ax.hlines(heights[0], 
				leftIndex[0], 
				rightIndex[0], 
				color='tab:olive', 
				linestyle = '-',
				label='FWHM')
		ax.hlines(heights[0] - 5, 
					peaks[0], 
					midpoints[0], 
					color='tab:orange', 
					linestyle = '-',
					label='Peak offset')
		ax.vlines(peaks[0], 
				smoothed_signal[peaks[0]]-proms[0],
				smoothed_signal[peaks[0]], 
				color='tab:purple', 
				linestyle = '-',
				label = 'Peak amplitude')
		
		ax.legend(loc='upper right', fontsize='small', ncol=1)
		ax.set_xlabel('Time (frames)')
		ax.set_ylabel('Signal (AU)')
		ax.set_title(f'{Ch_name} peak properties')
	plt.close(fig)

	return fig

def plot_indv_acf_workflow(
	bin_values: np.ndarray,
	indv_acfs: np.ndarray,
	img_parameters_dict: dict,
	img_props: dict
):
	num_channels = img_props['num_channels']
	num_bins = img_props['num_bins']
	num_frames = img_props['num_frames']
	indv_periods = img_parameters_dict['Period']
	analysis_type = img_props['analysis_type']

	indv_acf_plots = {}
	its = num_channels*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind acfs')
		for channel in range(num_channels):
			for bin in range(num_bins):
				pbar.update(1) 
				to_plot = bin_values[:,channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
				indv_acf_plots[f'Ch{channel + 1} Bin {bin + 1} ACF'] = return_indv_acf_figure(
					raw_signal=to_plot, 
					acf_curve=indv_acfs[channel, bin], 
					Ch_name=f'Ch{channel + 1}', 
					period=indv_periods[channel, bin],
					num_frames=num_frames
					)
	return indv_acf_plots

def return_indv_acf_figure(
	raw_signal: np.ndarray, 
	acf_curve: np.ndarray, 
	Ch_name: str, 
	period: int,
	num_frames: int) -> plt.Figure:

	# Create subplots for raw signal and autocorrelation curve
	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(raw_signal)
	ax1.set_xlabel(f'{Ch_name} Raw Signal')
	ax1.set_ylabel('Mean bin px value')
	ax2.plot(np.arange(-num_frames + 1, num_frames), acf_curve)
	ax2.set_ylabel('Autocorrelation')

	# Annotate the first peak identified as the period if available
	if not period == np.nan:
			color = 'red'
			ax2.axvline(x = period, alpha = 0.5, c = color, linestyle = '--')
			ax2.axvline(x = -period, alpha = 0.5, c = color, linestyle = '--')
			ax2.set_xlabel(f'Period is {period} frames')
	else:
			ax2.set_xlabel(f'No period identified')

			fig.subplots_adjust(hspace=0.5)
			plt.close(fig)

	return(fig)

def plot_indv_ccf_workflow(
	bin_values: np.ndarray,
	indv_ccfs: np.ndarray,
	img_parameters_dict: dict,
	img_props: dict
) -> dict:
	channel_combos = img_props['channel_combos']
	num_bins = img_props['num_bins']
	num_frames = img_props['num_frames']
	indv_shifts = img_parameters_dict['Shift']
	analysis_type = img_props['analysis_type']
	
	indv_ccf_plots = {}
	its = len(channel_combos)*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind ccfs')
		for combo_number, combo in enumerate(channel_combos):
			for bin in range(num_bins):
				pbar.update(1)
				if analysis_type == 'standard':
					to_plot1 = bin_values[:, combo[0], bin] 
					to_plot2 = bin_values[:, combo[1], bin] 
				else:
					to_plot1 = bin_values[combo[0], bin]
					to_plot2 = bin_values[combo[1], bin]
				indv_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = return_indv_ccf_figure(
					ch1 = normalize_signal(to_plot1),
					ch2 = normalize_signal(to_plot2),
					ccf_curve = indv_ccfs[combo_number, bin],
					ch1_name = f'Ch{combo[0] + 1}',
					ch2_name = f'Ch{combo[1] + 1}',
					shift = indv_shifts[combo_number, bin],
					num_frames = num_frames)
				
	return indv_ccf_plots

def return_indv_ccf_figure(
	ch1: np.ndarray, 
	ch2: np.ndarray, 
	ccf_curve: np.ndarray, 
	ch1_name: str, 
	ch2_name: str, 
	shift: int,
	num_frames: int
) -> plt.Figure:
	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(ch1, color = 'tab:blue', label = ch1_name)
	ax1.plot(ch2, color = 'tab:orange', label = ch2_name)
	ax1.set_xlabel('time (frames)')
	ax1.set_ylabel('Mean bin px value')
	ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
	ax2.plot(np.arange(-num_frames + 1, num_frames), ccf_curve)
	ax2.set_ylabel('Crosscorrelation')
	
	# Annotate the first peak identified as the shift if available
	if not shift == np.nan:
		color = 'red'
		ax2.axvline(x = shift, alpha = 0.5, c = color, linestyle = '--')
		if shift < 1:
			ax2.set_xlabel(f'{ch1_name} leads by {int(abs(shift))} frames')
		elif shift > 1:
			ax2.set_xlabel(f'{ch2_name} leads by {int(abs(shift))} frames')
		else:
			ax2.set_xlabel('no shift detected')
	else:
		ax2.set_xlabel(f'No peaks identified')
	
	fig.subplots_adjust(hspace=0.5)
	plt.close(fig)

	return(fig)