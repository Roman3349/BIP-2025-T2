import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt, find_peaks

# Define the measurement directory and output directory
measurement_dir = Path("data")
output_dir = Path("plots")
for file in output_dir.rglob("*.png"):
    os.remove(file)
# Create an output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# Sampling frequency and Butterworth 4th order filter parameters
fs = 60
cutoff = 8
order = 4
b, a = butter(
    N=order, # Filter order
    Wn=[0.05  / (0.5 * fs), cutoff / (0.5 * fs)], # Nyquist (normalized) frequency
    btype='bandpass', # Band-pass filter
    analog=False, # Digital filter
    output='ba', # Output type - numerator and denominator
)

# Sensor colors and labels in the plot
sensor_colors = {
    '2': 'blue',
    '3': 'red',
    '4': 'green',
    '5': 'orange',
}
sensor_labels = {
    '2': 'Thigh',
    '3': 'Shank',
    '4': 'Foot',
    '5': 'Skateboard deck',
}

axes = [
    'Euler_Y',
    'FreeAcc_X', 'FreeAcc_Z',
    'Velocity_X', 'Velocity_Z',
]


def load_sensor_csv(filepath) -> pd.DataFrame:
    """
    Load the sensor data from a CSV file, process the time and acceleration data.
    :param filepath: Path to the CSV file
    :return: DataFrame with processed data
    """
    df: pd.DataFrame = pd.read_csv(filepath, sep=',', skiprows=11).drop_duplicates(subset='SampleTimeFine')
    ticks = df['SampleTimeFine'].astype('uint64').values
    # Adjust for time rollover which happens when the tick count exceeds 2^32
    for i in range(1, len(ticks)):
        if ticks[i] < ticks[i - 1]:
            ticks[i:] += 2**32

    df['SampleTimeFine'] = ticks
    # Convert ticks to seconds and set as index
    df['Time_s'] = (df['SampleTimeFine'] - df['SampleTimeFine'].min()) * 1e-6
    df = df[[
        'Euler_X', 'Euler_Y', 'Euler_Z',
        'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z',
        'Time_s',
    ]].set_index('Time_s').apply(lambda x: filtfilt(b, a, x), axis=0)
    # Fix axes
    sensor_id = filepath.stem.split('-')[0]
    if sensor_id in ['3', '4']:
        df[['FreeAcc_X', 'FreeAcc_Z']] = df[['FreeAcc_Z', 'FreeAcc_X']].copy()
    side = get_side(filepath.parent.name)
    if side == 'right':
        df['Euler_Y'] *= -1
    # Calculate velocity
    for axis in ['X', 'Y', 'Z']:
        df[f'Velocity_{axis}'] = cumulative_trapezoid(df[f'FreeAcc_{axis}'], dx=(1 / fs), initial=0)
        b1, a1 = butter(
            N=order,  # Filter order
            Wn=0.25 / (0.5 * fs),  # Nyquist (normalized) frequency
            btype='high',  # High-pass filter
            analog=False,  # Digital filter
            output='ba',  # Output type - numerator and denominator
        )
        df[f'Velocity_{axis}'] = filtfilt(b1, a1, df[f'Velocity_{axis}'])

    return df[[
        'Euler_X', 'Euler_Y', 'Euler_Z',
        'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z',
        'Velocity_X', 'Velocity_Y', 'Velocity_Z',
    ]]


def get_side(folder_name) -> str:
    trial = folder_name.split('_')[-1]
    if trial == 'R1' or trial == 'R2':
        return "right"
    elif trial == 'L1' or trial == 'L2':
        return "left"
    return "unknown"

def get_trial_type(folder_name) -> str:
    """
    Determine the trial type based on the folder name.
    :param folder_name: Name of the folder containing the trial data
    :return: String representing the trial type
    """
    trial = folder_name.split('_')[-1]
    if trial == 'R1':
        return "Right leg #1"
    if trial == 'R2':
        return "Right leg #2"
    if trial == 'L1':
        return "Left leg #1"
    if trial == 'L2':
        return "Left leg #2"
    return "Unknown type"


def find_gaps(time_index, max_gap_sec=0.2) -> list[tuple[int, int]]:
    """
    Find gaps in the time index where the difference between consecutive timestamps exceeds max_gap_sec.
    :param time_index: Timestamp index of the DataFrame
    :param max_gap_sec: Maximum allowed gap in seconds
    :return: List of tuples representing the start and end of each gap
    """
    gaps: list[tuple[int, int]] = []
    # Convert the time index to a numpy array for easier manipulation
    t: np.array = time_index.to_numpy()
    # Calculate the time differences between consecutive timestamps
    dt = t[1:] - t[:-1]
    gap_indices = (dt > max_gap_sec).nonzero()[0]
    for idx in gap_indices:
        gaps.append((t[idx], t[idx + 1]))
    return gaps


def detect_peaks(time, signal, height=5.0, min_height=2.0, min_time=0.4) -> list[tuple[int, int]]:
    """
    Detects peaks in the signal using the find_peaks function from scipy.
    :param time: Time axis
    :param signal: Signal data
    :param height: Minimum height of the peaks
    :param min_height: Absolute minimum height of the peaks
    :param min_time: Minimum time between peaks
    :return: List of tuples containing the time and value of each detected peak
    """
    if height < 0:
        signal_copy = signal.copy() * -1
        height *= -1
    else:
        signal_copy = signal.copy()
    height = max(height, min_height)
    peaks, properties = find_peaks(signal_copy, height=height, distance=min_time * fs, prominence=2.5 * height)
    return [(time[i], signal_copy[i] * (-1 if height < 0 else 1)) for i in peaks]

def plot_peaks(plot_axis, peaks: list[tuple[int, int]], plot_point: bool, color: str) -> None:
    """
    Plot the detected peaks on the given axis.
    :param plot_axis: Plot axis to draw on
    :param peaks: List of tuples containing the time and value of each detected peak
    :param plot_point: Whether to plot the peak points
    :param color: Color for the peaks
    """
    for peak_time, peak_value in peaks:
        if plot_point:
            plot_axis.plot(peak_time, peak_value, f'{color[0]}o', markersize=4)
        # Add a vertical line at the step time
        plot_axis.axvline(x=peak_time, color=color, linestyle='--', alpha=0.5)
        plot_axis.axvspan(peak_time - 0.05, peak_time + 0.05, color=color, alpha=0.1)


sensor_data = {}
stats = []

# Iterate through all CSV files in the measurement directory
for file_path in measurement_dir.rglob("*.csv"):
    try:
        print(f"Processing {file_path}")
        # Load and filter the sensor data
        df = load_sensor_csv(file_path)
        # Parse the sensor ID and trial information from the file path
        sensor_id = file_path.stem.split('-')[0]
        trial_type = get_trial_type(file_path.parent.name)
        trial_id = file_path.parent.name.split('_')[1]
        if trial_type == "Unknown type":
            trial_type += f" - {trial_id}"
        subject_id = file_path.parent.parent.name[0:3]
        trial_name = f"Subject {subject_id} - trial {trial_type}"
        sensor_data.setdefault(trial_name, {})[sensor_id] = {
            'data': df,
            'trial_type': trial_type,
            'side': 'right' if trial_type.startswith('Right') else 'left',
            'label': sensor_labels.get(sensor_id, f"Sensor {sensor_id}"),
            'color': sensor_colors.get(sensor_id, 'black')
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

for trial_name, sensors in sensor_data.items():
    if '4' not in sensors:
        print(f"Skipping {trial_name} - no foot sensor")
        continue
    if '3' not in sensors:
        print(f"Skipping {trial_name} - no shank sensor")
        continue
    # Detect first 10 pushes in FreeAcc_Z
    sensor_values = sensors['4']['data']
    pushes = detect_peaks(
        time=sensor_values.index.to_numpy(),
        signal=sensor_values['FreeAcc_Z'].to_numpy(),
        height=min(sensor_values['FreeAcc_Z']) * 0.15,
        min_height=2.0,
    )[1:8]
    # Detect lift off in Euler_Y
    sensor_values = sensors['3']['data']
    lift_offs_orig = detect_peaks(
        time=sensor_values.index.to_numpy(),
        signal=sensor_values['Euler_Y'].to_numpy(),
        height=max(sensor_values['Euler_Y']) * 0.3,
        min_height=5.0,
    )
    lift_offs = []
    # Remove liftoffs that are before the first push
    for lift_off_time, lift_off_value in lift_offs_orig:
        if pushes[0][0] < lift_off_time < pushes[-1][0]:
            lift_offs.append((lift_off_time, lift_off_value))

    for axis in [
        'Euler_Y',
        'FreeAcc_X', 'FreeAcc_Z',
        'Velocity_X', 'Velocity_Z',
    ]:
        print(f"Plotting {trial_name} - {axis}")
        fig, axs = plt.subplots(len(sensors), 1, figsize=(16, 2.5 * len(sensors)), sharex=True)

        # If only one sensor, axs is not a list
        if len(sensors) == 1:
            axs = [axs]

        for ax, (sensor_id, sensor_info) in zip(axs, sorted(sensors.items())):
            data = sensor_info['data']
            limit_low = max(pushes[0][0] - 1.5, 0)
            limit_high = min(pushes[-1][0] + 1.5, data.index[-1])
            data = data[(data.index >= limit_low) & (data.index <= limit_high)]
            # Time axis
            t = data.index.to_numpy()
            # Acceleration axis
            y: np.ndarray = data[axis].to_numpy()

            # Find gaps in the data
            gaps = find_gaps(data.index)

            # Split the data into segments based on gaps
            segments = []
            last_idx = 0
            for gap_start, gap_end in gaps:
                split_idx = (t > gap_start)[0:].argmax()
                segments.append((t[last_idx:split_idx], y[last_idx:split_idx]))
                last_idx = split_idx
                print(f"Gap found: {gap_start} to {gap_end}, trial {trial_name}, sensor {sensor_id}")
            segments.append((t[last_idx:], y[last_idx:]))

            # Plot the segments
            for t_seg, y_seg in segments:
                if len(t_seg) == len(y_seg) and len(t_seg) > 1:
                    ax.plot(t_seg, y_seg, color=sensor_info['color'])
            # Plot the gaps
            for gap_start, gap_end in gaps:
                ax.axvspan(gap_start, gap_end, color='red', alpha=0.3)

            # Plot pushes in FreeAcc_* except FreeAcc_Total
            if len(y) > 0:
                plot_peaks(ax, pushes, axis == "FreeAcc_Z" and sensor_id == 4, 'green')
                plot_peaks(ax, lift_offs, axis == "Euler_Y" and sensor_id == 3, 'red')

                if axis in ['Euler_Y', 'FreeAcc_X', 'FreeAcc_Z']:
                    duration = pushes[-1][0] - pushes[0][0]
                    stats.append({
                        'trial_name': trial_name,
                        'label': sensor_info['label'],
                        'axis': axis,
                        'min': min(y),
                        'mean': y.mean(),
                        'max': max(y),
                        'duration': duration,
                        'pushes': len(pushes),
                        'lift_offs': len(lift_offs),
                        'cadence': len(pushes) / duration,
                    })
            unit_label = 'unknown [-]'
            if axis.startswith("FreeAcc_"):
                unit_label = 'acceleration [m/s²]'
            elif axis.startswith("Velocity_"):
                unit_label = 'velocity [m/s]'
            elif axis.startswith("Euler_"):
                unit_label = 'euler angle [°]'
            # Set the title and labels for each subplot
            ax.set_ylabel(sensor_info['label'] + "\n" + unit_label, fontsize=10)
            # Show grid for better readability
            ax.grid(True)

        # Set the x-axis label for the last subplot
        axs[-1].set_xlabel("Time [s]")
        # Set the title for the entire figure
        fig.suptitle(f"{trial_name} - {"Axis: " + axis}", fontsize=14)
        # Adjust the layout to prevent overlap
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        # Save the figure to the output directory
        output_path = output_dir / f"{trial_name.replace(' ', '_')}_{axis}.png"
        fig.savefig(output_path)
        plt.close(fig)

with open("stats.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=stats[0].keys())
    writer.writeheader()
    writer.writerows(stats)
