import csv
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Define the measurement directory and output directory
root_dir = Path(".")
output_dir = Path("plots")
# Create an output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# Sampling frequency and Butterworth 4th order filter parameters
fs = 60
cutoff = 2
order = 4
b, a = butter(
    N=order, # Filter order
    Wn=cutoff / (0.5 * fs), # Nyquist (normalized) frequency
    btype='low', # Low-pass filter
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
    df = df.set_index('Time_s')

    return df[['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z']]


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


def detect_kicks_with_peaks(t, acc_y, height=5.0, time=0.5):
    peaks, properties = find_peaks(acc_y, height=height, distance=time * fs)
    kicks = [(t[i], acc_y[i]) for i in peaks]
    return kicks


sensor_data = {}

# Iterate through all CSV files in the measurement directory
for file_path in root_dir.rglob("Xsens_DOT_*.csv"):
    try:
        print(f"Processing {file_path}")
        # Load and filter the sensor data
        df = load_sensor_csv(file_path)
        df_filtered = df.apply(lambda x: filtfilt(b, a, x), axis=0)
        # Calculate the total free acceleration
        df_filtered['FreeAcc_Total'] = (df_filtered[['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z']] ** 2).sum(axis=1).pow(0.5)

        sensor_id = file_path.stem.split('_')[2]
        if sensor_id in ['3', '4']:
            df_filtered[['FreeAcc_X', 'FreeAcc_Z']] = df_filtered[['FreeAcc_Z', 'FreeAcc_X']].copy()

        trial_type = get_trial_type(file_path.parent.name)
        subject_id = file_path.parent.name.split('_')[0]
        trial_name = f"Subject {subject_id} - {trial_type}"

        sensor_data.setdefault(trial_name, {})[sensor_id] = {
            'data': df_filtered,
            'trial_type': trial_type,
            'label': sensor_labels.get(sensor_id, f"Sensor {sensor_id}") + "\nfree acceleration [m/s²]",
            'color': sensor_colors.get(sensor_id, 'black')
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

for trial_name, sensors in sensor_data.items():
    for axis in ['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z', 'FreeAcc_Total']:
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

            # Plot kicks
            if sensor_labels.get(sensor_id, '') in ['Right leg', 'Left leg'] and (axis == "FreeAcc_X" or axis == "FreeAcc_Y"):
                kicks = detect_kicks_with_peaks(t, y, max(max(y) * 0.33, (5.0 if axis == "FreeAcc_Y" else 2.0)))
                for peak_time, peak_value in kicks:
                    ax.axvspan(peak_time - 0.1, peak_time + 0.1, color='blue', alpha=0.2)

                if sensor_info['trial_type'] == "Alternating legs - uphill" and axis == "FreeAcc_X" or sensor_info['trial_type'] != "Alternating legs - uphill" and axis == "FreeAcc_Y":
                    sensors[sensor_id]['max_acc'] = max(y)
                    sensors[sensor_id]['kick_count'] = len(kicks)
                    sensors[sensor_id]['duration'] = t[-1] - t[0]
                    sensors[sensor_id]['kicks'] = kicks

            # Set the title and labels for each subplot
            ax.set_ylabel(sensor_info['label'])
            # Show grid for better readability
            ax.grid(True)

        # Set the x-axis label for the last subplot
        axs[-1].set_xlabel("Time [s]")
        # Set the title for the entire figure
        fig.suptitle(f"{trial_name} - {"Axis: " + axis[-1] if axis != 'FreeAcc_Total' else 'Total'}", fontsize=14)
        # Adjust the layout to prevent overlap
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        # Save the figure to the output directory
        output_path = output_dir / f"{trial_name.replace(' ', '_')}_{axis[-1] if axis != 'FreeAcc_Total' else 'Total'}.png"
        fig.savefig(output_path)
        plt.close(fig)
        
        
results = []
for trial_name, sensors in sensor_data.items():
    left_id = next((k for k in sensors if sensor_labels.get(k, '') == 'Left leg'), None)
    right_id = next((k for k in sensors if sensor_labels.get(k, '') == 'Right leg'), None)

    if left_id and right_id:
        left_kicks = sensors[left_id].get('kick_count', 0)
        right_kicks = sensors[right_id].get('kick_count', 0)
        left_dur = sensors[left_id].get('duration', 1)
        right_dur = sensors[right_id].get('duration', 1)
        left_max_acc = sensors[left_id].get('max_acc', 0)
        right_max_acc = sensors[right_id].get('max_acc', 0)

        left_freq = left_kicks / left_dur
        right_freq = right_kicks / right_dur
        acc_ratio = right_max_acc / (left_max_acc + right_max_acc) if (left_kicks + right_kicks) > 0 else 0.5

        results.append({
            'trial': trial_name,
            'left_kicks': left_kicks,
            'right_kicks': right_kicks,
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
