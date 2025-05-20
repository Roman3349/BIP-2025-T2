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
cutoff = 2
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
    '1': 'blue',
    '2': 'red',
    '3': 'green',
    '4': 'orange',
}
sensor_labels = {
    '2': 'Thigh',
    '3': 'Shank',
    '4': 'Foot',
    '5': 'Skateboard deck',
}


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
    df = df.set_index('Time_s')
    # Fix axes
    sensor_id = filepath.stem.split('-')[0]
    if sensor_id in ['3', '4']:
        df[['FreeAcc_X', 'FreeAcc_Z']] = df[['FreeAcc_Z', 'FreeAcc_X']].copy()

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


def detect_kicks_with_peaks(t, acc_y, height=5.0, time=0.5):
    peaks, properties = find_peaks(acc_y, height=height, distance=time * fs)
    kicks = [(t[i], acc_y[i]) for i in peaks]
    return kicks


sensor_data = {}

# Iterate through all CSV files in the measurement directory
for file_path in measurement_dir.rglob("*.csv"):
    try:
        print(f"Processing {file_path}")
        # Load and filter the sensor data
        df = load_sensor_csv(file_path)
        df_filtered = df.apply(lambda x: filtfilt(b, a, x), axis=0)
        # Parse the sensor ID and trial information from the file path
        sensor_id = file_path.stem.split('-')[0]
        trial_type = get_trial_type(file_path.parent.name)
        trial_id = file_path.parent.name.split('_')[1]
        if trial_type == "Unknown type":
            trial_type += f" - {trial_id}"
        subject_id = file_path.parent.parent.name[0:3]
        trial_name = f"Subject {subject_id} - trial {trial_type}"
        unit = "unknows [-]"
        sensor_data.setdefault(trial_name, {})[sensor_id] = {
            'data': df_filtered,
            'trial_type': trial_type,
            'label': sensor_labels.get(sensor_id, f"Sensor {sensor_id}"),
            'color': sensor_colors.get(sensor_id, 'black')
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

for trial_name, sensors in sensor_data.items():
    for axis in [
        'Euler_X', 'Euler_Y', 'Euler_Z',
        'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z',
        'Velocity_X', 'Velocity_Y', 'Velocity_Z',
    ]:
        print(f"Plotting {trial_name} - {axis}")
        fig, axs = plt.subplots(len(sensors), 1, figsize=(10, 2.5 * len(sensors)), sharex=True)

        # If only one sensor, axs is not a list
        if len(sensors) == 1:
            axs = [axs]

        for ax, (sensor_id, sensor_info) in zip(axs, sorted(sensors.items())):
            data = sensor_info['data']
            # Time axis
            t = data.index.to_numpy()
            # Acceleration axis
            y = data[axis].to_numpy()

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

            # Plot kicks in FreeAcc_* except FreeAcc_Total
            if axis.startswith("FreeAcc_") and axis != "FreeAcc_Total" and len(y) > 0:
                kicks = detect_kicks_with_peaks(t, y, max(max(y) * 0.33, (0.5 if sensor_id == 4 else 2.0)))
                for peak_time, peak_value in kicks:
                    ax.plot(peak_time, peak_value, 'bo', markersize=4)
                    # Add a vertical line at the step time
                    ax.axvline(x=peak_time, color='blue', linestyle='--', alpha=0.5)
                    ax.axvspan(peak_time - 0.1, peak_time + 0.1, color='blue', alpha=0.1)

                if axis == "FreeAcc_X":
                    sensors[sensor_id]['axis'] = axis
                    sensors[sensor_id]['max_acc'] = max(y)
                    sensors[sensor_id]['kick_count'] = len(kicks)
                    sensors[sensor_id]['duration'] = t[-1] - t[0]
                    sensors[sensor_id]['kicks'] = kicks

            unit_label = 'unknown [-]'
            if axis.startswith("FreeAcc_"):
                unit_label = 'acceleration [m/s²]'
            elif axis.startswith("Velocity_"):
                unit_label = 'velocity [m/s]'
            elif axis.startswith("Euler_"):
                unit_label = 'angular velocity [°/s]'
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

