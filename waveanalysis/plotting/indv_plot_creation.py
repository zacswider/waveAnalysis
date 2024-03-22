import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from signal_processing import normalize_signal

def plot_indv_peak_workflow(
	bin_values: np.ndarray,
	img_prop_dict: dict,
	indv_peak_props: dict,
	num_frames: int
) -> dict:
	"""
	Generates individual peak plots for each channel and bin.

	Args:
		bin_values (np.ndarray): Array of bin values.
		img_prop_dict (dict): Dictionary containing image properties.
		indv_peak_props (dict): Dictionary containing individual peak properties.
		num_frames (int): Number of frames.

	Returns:
		dict: Dictionary containing the generated individual peak plots.
	"""
	# Extract image properties from the dictionary
	num_channels = img_prop_dict['num_channels']
	num_bins = img_prop_dict['num_bins']
	analysis_type = img_prop_dict['analysis_type']
	frame_interval = img_prop_dict['frame_interval']

	# Initialize dictionary to store the individual peak plots
	indv_peak_figs = {}
	
	# Loop through each channel and bin
	its = num_channels*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind peaks')
		for channel in range(num_channels):
			for bin in range(num_bins):
				pbar.update(1)
				# Extract the bin values for the current channel and bin
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
	'''
	Space saving function to return individual peak property figures
	'''
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
			# Plot the peal width
			ax.hlines(heights[i], 
					leftIndex[i] * frame_interval, 
					rightIndex[i] * frame_interval, 
					color='tab:olive', 
					linestyle = '-')
			# Plot the peak amplitude
			ax.vlines(peaks[i] * frame_interval, 
					smoothed_signal[peaks[i]]-proms[i],
					smoothed_signal[peaks[i]], 
					color='tab:purple', 
					linestyle = '-')
			# Plot the peak offset
			ax.hlines(heights[i]-5, 
					peaks[i] * frame_interval, 
					midpoints[i] * frame_interval, 
					color='tab:orange', 
					linestyle = '-')

		# Plot the legend for the first peak
		ax.hlines(heights[0], 
				leftIndex[0] * frame_interval, 
				rightIndex[0] * frame_interval, 
				color='tab:olive', 
				linestyle = '-',
				label='FWHM')
		ax.vlines(peaks[0] * frame_interval, 
				smoothed_signal[peaks[0]]-proms[0],
				smoothed_signal[peaks[0]], 
				color='tab:purple', 
				linestyle = '-',
				label = 'Peak amplitude')
		ax.hlines(heights[0] - 5, 
					peaks[0] * frame_interval, 
					midpoints[0] * frame_interval, 
					color='tab:orange', 
					linestyle = '-',
					label='Peak offset')
		
		# Plot the legend for the rest of the peaks
		ax.legend(loc='upper right', fontsize='small', ncol=1)
		ax.set_xlabel('Time (seconds)')
		ax.set_ylabel('Signal (AU)')
		ax.set_title(f'{Ch_name} peak properties')
	plt.close(fig)

	return fig

def plot_indv_acf_workflow(
	bin_values: np.ndarray,
	indv_acfs: np.ndarray,
	img_parameters_dict: dict,
	img_props: dict
) -> dict:
	"""
	Generates individual ACF plots for each channel and bin.

	Args:
		bin_values (np.ndarray): Array of bin values.
		indv_acfs (np.ndarray): Array of individual ACF values.
		img_parameters_dict (dict): Dictionary of image parameters.
		img_props (dict): Dictionary of image properties.

	Returns:
		dict: Dictionary containing individual ACF plots.
	"""
	# Extract image properties from the dictionary
	num_channels = img_props['num_channels']
	num_bins = img_props['num_bins']
	num_frames = img_props['num_frames']
	indv_periods = img_parameters_dict['Period']
	analysis_type = img_props['analysis_type']
	frame_interval = img_props['frame_interval']

	# Initialize dictionary to store the individual ACF plots
	indv_acf_plots = {}

	# Loop through each channel and bin
	its = num_channels*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind acfs')
		for channel in range(num_channels):
			for bin in range(num_bins):
				pbar.update(1) 
				# Extract the bin values for the current channel and bin
				to_plot = bin_values[:,channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
				# Generate and store the figure for the current channel and bin
				indv_acf_plots[f'Ch{channel + 1} Bin {bin + 1} ACF'] = return_indv_acf_figure(
					raw_signal=to_plot, 
					acf_curve=indv_acfs[channel, bin], 
					Ch_name=f'Ch{channel + 1}', 
					period=indv_periods[channel, bin],
					num_frames=num_frames,
					frame_interval=frame_interval
					)
				
	return indv_acf_plots

def return_indv_acf_figure(
	raw_signal: np.ndarray, 
	acf_curve: np.ndarray, 
	Ch_name: str, 
	period: int,
	num_frames: int,
	frame_interval: float
) -> plt.Figure:
	'''
	Space saving function to return individual ACF figures
	'''
	# Create subplots for raw signal and autocorrelation curve
	fig, (ax1, ax2) = plt.subplots(2, 1)
	x_axis = np.arange(0, num_frames) * frame_interval
	# Plot the raw signal and autocorrelation curve
	ax1.plot(x_axis, raw_signal)
	ax1.set_xlabel(f'{Ch_name} Raw Signal')
	ax1.set_ylabel('Mean bin px value')
	# Plot the autocorrelation curve
	ax2.plot(np.arange(-num_frames + 1, num_frames) * frame_interval, acf_curve)
	ax2.set_ylabel('Autocorrelation')

	# Annotate the first peak identified as the period if available
	if not period == np.nan:
			color = 'red'
			ax2.axvline(x = period, alpha = 0.5, c = color, linestyle = '--')
			ax2.axvline(x = -period, alpha = 0.5, c = color, linestyle = '--')
			ax2.set_xlabel(f'Period is {abs(round(period, 2))} seconds')
	else:
			ax2.set_xlabel(f'No period identified')

	fig.subplots_adjust(hspace=0.75)
	plt.close(fig)

	return(fig)

def plot_indv_ccf_workflow(
	bin_values: np.ndarray,
	indv_ccfs: np.ndarray,
	img_parameters_dict: dict,
	img_props: dict
) -> dict:
	"""
	Plot individual cross-correlation function (CCF) workflow.

	Parameters:
	- bin_values (np.ndarray): Array of bin values.
	- indv_ccfs (np.ndarray): Array of individual CCFs.
	- img_parameters_dict (dict): Dictionary of image parameters.
	- img_props (dict): Dictionary of image properties.

	Returns:
	- indv_ccf_plots (dict): Dictionary of individual CCF plots.
	"""
	# Extract image properties from the dictionary
	channel_combos = img_props['channel_combos']
	num_bins = img_props['num_bins']
	num_frames = img_props['num_frames']
	indv_shifts = img_parameters_dict['Shift']
	analysis_type = img_props['analysis_type']
	frame_interval = img_props['frame_interval']

	# Initialize dictionary to store the individual CCF plots
	indv_ccf_plots = {}

	# Loop through each channel and bin
	its = len(channel_combos)*num_bins
	with tqdm(total=its, miniters=its/100) as pbar:
		pbar.set_description('ind ccfs')
		for combo_number, combo in enumerate(channel_combos):
			for bin in range(num_bins):
				pbar.update(1)
				# Extract the bin values for the current channel and bin
				if analysis_type == 'standard':
					to_plot1 = bin_values[:, combo[0], bin] 
					to_plot2 = bin_values[:, combo[1], bin] 
				else:
					to_plot1 = bin_values[combo[0], bin]
					to_plot2 = bin_values[combo[1], bin]
				# Generate and store the figure for the current channel and bin
				indv_ccf_plots[f'Ch{combo[0]}-Ch{combo[1]} Bin {bin + 1} CCF'] = return_indv_ccf_figure(
					ch1 = normalize_signal(to_plot1),
					ch2 = normalize_signal(to_plot2),
					ccf_curve = indv_ccfs[combo_number, bin],
					ch1_name = f'Ch{combo[0] + 1}',
					ch2_name = f'Ch{combo[1] + 1}',
					shift = indv_shifts[combo_number, bin],
					num_frames = num_frames,
					frame_interval = frame_interval)
				
	return indv_ccf_plots

def return_indv_ccf_figure(
	ch1: np.ndarray, 
	ch2: np.ndarray, 
	ccf_curve: np.ndarray, 
	ch1_name: str, 
	ch2_name: str, 
	shift: int,
	num_frames: int,
	frame_interval: float
) -> plt.Figure:
	'''
	Space saving function to return individual CCF figures
	'''
	fig, (ax1, ax2) = plt.subplots(2, 1)
	x_axis = np.arange(0, num_frames) * frame_interval 
	# Plot the raw signal
	ax1.plot(x_axis, ch1, color = 'tab:blue', label = ch1_name)
	ax1.plot(x_axis, ch2, color = 'tab:orange', label = ch2_name)
	ax1.set_xlabel('time (seconds)')
	ax1.set_ylabel('Mean bin px value')
	# Plot the autocorrelation curve
	ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
	ax2.plot(np.arange(-num_frames + 1, num_frames) * frame_interval, ccf_curve)
	ax2.set_ylabel('Crosscorrelation')
	
	# Annotate the first peak identified as the shift if available
	if not shift == np.nan:
		color = 'red'
		ax2.axvline(x = shift, alpha = 0.5, c = color, linestyle = '--')
		if shift < 1:
			ax2.set_xlabel(f'{ch1_name} leads by {abs(round(shift, 2))} seconds')
		elif shift > 1:
			ax2.set_xlabel(f'{ch2_name} leads by {abs(round(shift, 2))} seconds')
		else:
			ax2.set_xlabel('no shift detected')
	else:
		ax2.set_xlabel(f'No peaks identified')
	
	fig.subplots_adjust(hspace=0.5)
	plt.close(fig)

	return(fig)