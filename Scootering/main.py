import csv
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.transform import Rotation

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
    '5': 'red',
}
sensor_labels = {
    '1': 'Right leg',
    '2': 'Left leg',
    '3': 'Lower back',
    '4': 'Scooter',
    '5': 'Left leg',
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
    if 'Quat_W' in df.columns:
        quaternion = df[['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']].values
        euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
        df['Euler_X'] = euler[:, 0]
        df['Euler_Y'] = euler[:, 1]
        df['Euler_Z'] = euler[:, 2]
    df = df[[
        'Euler_X', 'Euler_Y', 'Euler_Z',
        'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z',
        'Time_s',
    ]].set_index('Time_s').apply(lambda x: filtfilt(b, a, x), axis=0)
    # Fix axes
    sensor_id = filepath.stem.split('-')[0]
    if sensor_id == 5:
        sensor_id = 2
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
    if "only_right_flat" in folder_name:
        return "Only right leg - flat"
    elif "only_left_flat" in folder_name:
        return "Only left leg - flat"
    elif "alternating_flat" in folder_name:
        return "Alternating legs - flat"
    elif "alternating_hill" in folder_name:
        return "Alternating legs - uphill"
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


def detect_pushes_with_peaks(t, acc_y, height=5.0, time=0.5):
    peaks, properties = find_peaks(acc_y, height=height, distance=time * fs)
    pushes = [(t[i], acc_y[i]) for i in peaks]
    return pushes


sensor_data = {}

# Iterate through all CSV files in the measurement directory
for file_path in measurement_dir.rglob("Xsens_DOT_*.csv"):
    try:
        print(f"Processing {file_path}")
        # Load and filter the sensor data
        df = load_sensor_csv(file_path)

        sensor_id = file_path.stem.split('_')[2]
        trial_type = get_trial_type(file_path.parent.name)
        subject_id = file_path.parent.name.split('_')[0]
        trial_name = f"Subject {subject_id} - {trial_type}"

        sensor_data.setdefault(trial_name, {})[sensor_id] = {
            'data': df,
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

            # Plot pushes in FreeAcc_* except FreeAcc_Total
            if (
                axis.startswith("FreeAcc_") and
                axis != "FreeAcc_Total" and
                len(y) > 0 and
                sensor_labels.get(sensor_id, '') in ['Right leg', 'Left leg']
            ):
                inverted = False
                if inverted:
                    pushes = detect_pushes_with_peaks(t, -1 * y, max(min(y) * -0.4, 2.0))
                else:
                    pushes = detect_pushes_with_peaks(t, y, max(max(y) * 0.4, 2.0))

                for peak_time, peak_value in pushes:
                    ax.plot(peak_time, (-1 if inverted else 1) * peak_value, 'bo', markersize=4)
                    # Add a vertical line at the step time
                    ax.axvline(x=peak_time, color='blue', linestyle='--', alpha=0.5)
                    ax.axvspan(peak_time - 0.1, peak_time + 0.1, color='blue', alpha=0.2)

                if axis == "FreeAcc_X":
                    sensors[sensor_id]['axis'] = axis
                    sensors[sensor_id]['max_acc'] = max(y)
                    sensors[sensor_id]['push_count'] = len(pushes)
                    sensors[sensor_id]['duration'] = t[-1] - t[0]
                    sensors[sensor_id]['pushes'] = pushes

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
        if axis.startswith('FreeAcc_'):
            subtitle = f'Acceleration in axis {axis[-1]}' 
        elif axis.startswith('Velocity_'):
            subtitle = f'Velocity in axis {axis[-1]}'
        elif axis.startswith('Euler_'):
            subtitle = f'Angle in axis {axis[-1]}'
        else:
            subtitle = f'Measurement in {axis}'

        fig.suptitle(f"{trial_name} - {subtitle}", fontsize=14)
        # Adjust the layout to prevent overlap
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        # Save the figure to the output directory
        output_path = output_dir / f"{trial_name.replace(' ', '_')}_{axis}.png"
        fig.savefig(output_path)
        plt.close(fig)

results = []
for trial_name, sensors in sensor_data.items():
    left_id = next((k for k in sensors if sensor_labels.get(k, '') == 'Left leg'), None)
    right_id = next((k for k in sensors if sensor_labels.get(k, '') == 'Right leg'), None)

    if left_id and right_id:
        left_pushes = sensors[left_id].get('push_count', 0)
        right_pushes = sensors[right_id].get('push_count', 0)
        left_dur = sensors[left_id].get('duration', 1)
        right_dur = sensors[right_id].get('duration', 1)
        left_max_acc = sensors[left_id].get('max_acc', 0)
        right_max_acc = sensors[right_id].get('max_acc', 0)

        left_freq = left_pushes / left_dur
        right_freq = right_pushes / right_dur
        acc_ratio = right_max_acc / (left_max_acc + right_max_acc) if (left_pushes + right_pushes) > 0 else 0.5

        results.append({
            'axis': sensors[left_id]['axis'],
            'trial': trial_name,
            'left_pushes': left_pushes,
            'right_pushes': right_pushes,
            'left_max_acc': round(left_max_acc, 2),
            'right_max_acc': round(right_max_acc, 2),
            'left_freq_hz': round(left_freq, 2),
            'right_freq_hz': round(right_freq, 2),
            'right_left_ratio': round(acc_ratio, 2)
        })

with open("stats.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
