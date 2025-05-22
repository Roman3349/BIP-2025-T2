import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
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
cutoff = 12
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
    if sensor_id == '3':
        df['Euler_Y'] *= -1
    subject_id = filepath.parent.parent.name.split('-')[0].strip()
    if subject_id in ['S03', 'S04', 'S06']:
        df['FreeAcc_X'] *= -1
    # Calculate velocity
    for axis in ['X', 'Y', 'Z']:
        df[f'Velocity_{axis}'] = cumulative_trapezoid(df[f'FreeAcc_{axis}'], dx=(1 / fs), initial=0)
        b1, a1 = butter(
            N=order,  # Filter order
            Wn=0.1 / (0.5 * fs),  # Nyquist (normalized) frequency
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


def detect_peaks(time, signal, height=5.0, min_height=2.0, min_time=None) -> list[tuple[int, int]]:
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
    peaks, properties = find_peaks(signal_copy, height=height, distance=min_time * fs if min_time is not None else None)
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
        #plot_axis.axvspan(peak_time - 0.05, peak_time + 0.05, color=color, alpha=0.1)


def plot_variable(ax, sensor_id: str, sensor_info: dict, axis: str, with_cropping: bool = True) -> None:
    """
    Plot the specified variable on the given axis.
    :param ax: Axis to plot on
    :param sensor_id: Sensor ID
    :param sensor_info: Dictionary containing sensor information
    :param axis: Variable to plot
    """
    data = sensor_info['data']
    if with_cropping:
        limit_low = max(groud_impacts[0][0] - 1, 0)
        limit_high = min(groud_impacts[-1][0] + 0.5, data.index[-1])
        data = data[(data.index >= limit_low) & (data.index <= limit_high)]
    # Time axis
    t = data.index.to_numpy()
    # Axis
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
    if len(y) > 0 and with_cropping:
        plot_peaks(ax, groud_impacts, axis == "FreeAcc_Z" and sensor_id == 4, 'green')
        plot_peaks(ax, lift_offs, axis == "Euler_Y" and sensor_id == 3, 'red')

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

def plot_cycles(cycles: dict) -> None:
    fig, axs = plt.subplots(4, 1, figsize=(16, 9), sharex=True)

    shank_angle_y: list[pd.DataFrame] = []
    foot_acceleration_x: list[pd.DataFrame] = []
    foot_acceleration_z: list[pd.DataFrame] = []
    skateboard_acceleration_x: list[pd.DataFrame] = []

    for cycle in cycles.values():
        new_time = np.linspace(0, 1, fs)
        shank_angle_y.append(pd.DataFrame({
            'time': new_time,
            'value': np.interp(new_time, cycle['3']['data'].index, cycle['3']['data']['Euler_Y']),
        }))
        foot_acceleration_x.append(pd.DataFrame({
            'time': new_time,
            'value': np.interp(new_time, cycle['4']['data'].index, cycle['4']['data']['FreeAcc_X']),
        }))
        foot_acceleration_z.append(pd.DataFrame({
            'time': new_time,
            'value': np.interp(new_time, cycle['4']['data'].index, cycle['4']['data']['FreeAcc_Z']),
        }))
        skateboard_acceleration_x.append(pd.DataFrame({
            'time': new_time,
            'value': np.interp(new_time, cycle['5']['data'].index, cycle['5']['data']['FreeAcc_X']),
        }))
    # Calculate mean of foot's FreeAcc_X
    foot_acceleration_x_concat = pd.concat(foot_acceleration_x, ignore_index=True)
    foot_acceleration_x_mean = foot_acceleration_x_concat.groupby('time').mean().reset_index()
    # Find one maximum peak and max peak location in mean foot's FreeAcc_X using find_peaks
    foot_acceleration_x_peak, peak_props = find_peaks(foot_acceleration_x_mean['value'])
    # Find the absolute maximum peak in foot's FreeAcc_X
    max_foot_acc_x_peak = None
    for i in foot_acceleration_x_peak:
        if max_foot_acc_x_peak is None or foot_acceleration_x_mean['value'][i] > foot_acceleration_x_mean['value'][max_foot_acc_x_peak]:
            max_foot_acc_x_peak = i
    max_foot_acc_x_peak = foot_acceleration_x_mean['time'][max_foot_acc_x_peak]
    sns.lineplot(ax=axs[0], data=pd.concat(shank_angle_y, ignore_index=True), x='time', y='value', errorbar='sd', color='red')
    sns.lineplot(ax=axs[1], data=pd.concat(foot_acceleration_x, ignore_index=True), x='time', y='value', errorbar='sd', color='green')
    sns.lineplot(ax=axs[2], data=pd.concat(foot_acceleration_z, ignore_index=True), x='time', y='value', errorbar='sd', color='blue')
    sns.lineplot(ax=axs[3], data=pd.concat(skateboard_acceleration_x, ignore_index=True), x='time', y='value', errorbar='sd', color='orange')
    axs[0].set_ylabel("Shank Y\neuler angle [°]", fontsize=10)
    axs[1].set_ylabel("Foot X\nacceleration [m/s²]", fontsize=10)
    axs[2].set_ylabel("Foot Z\nacceleration [m/s²]", fontsize=10)
    axs[3].set_ylabel("Skateboard X\nacceleration [m/s²]", fontsize=10)
    # Set the x-axis label for the last subplot
    axs[-1].set_xlabel("Normalized time [-]")
    for ax in axs:
        ax.grid(True, which='both', axis='both')
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.grid(which='major', linestyle='--', linewidth=0.5, color='gray')
    fig.suptitle(trial_name, fontsize=16)
    # Adjust the layout to prevent overlap
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    for ax in axs:
         ax.axvspan(0, max_foot_acc_x_peak, color='green', alpha=0.1)
         ax.axvspan(max_foot_acc_x_peak, 1, color='red', alpha=0.1)

    # Save the figure to the output directory
    output_path = output_dir / f"{trial_name.replace(' ', '_')}_cycle.png"
    fig.savefig(output_path)

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
    # Detect ground impacts in FreeAcc_Z
    sensor_values = sensors['4']['data']
    groud_impacts = detect_peaks(
        time=sensor_values.index.to_numpy(),
        signal=sensor_values['FreeAcc_X'].to_numpy(),
        height=min(sensor_values['FreeAcc_X']) * 0.2,
        min_height=2.0,
        min_time=0.5,
    )[2:8]
    # Detect lift offs in FreeAcc_X
    lift_offs = detect_peaks(
        time=sensor_values.index.to_numpy(),
        signal=sensor_values['FreeAcc_X'].to_numpy(),
        height=max(sensor_values['FreeAcc_X']) * 0.2,
        min_height=2.0,
        min_time=0.5,
    )[2:8]
    # Remove liftoffs that are before the first push
    for lift_off_time, lift_off_value in lift_offs:
        if not (groud_impacts[0][0] < lift_off_time < groud_impacts[-1][0]):
            lift_offs.remove((lift_off_time, lift_off_value))

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
            y = sensor_info['data'][axis]
            plot_variable(ax, sensor_id, sensor_info, axis)
            if len(y) > 0:
                if axis in ['Euler_Y', 'FreeAcc_X', 'FreeAcc_Z']:
                    duration = groud_impacts[-1][0] - groud_impacts[0][0]
                    stats.append({
                        'trial_name': trial_name,
                        'label': sensor_info['label'],
                        'axis': axis,
                        'min': min(y),
                        'mean': y.mean(),
                        'max': max(y),
                        'duration': duration,
                        'pushes': len(groud_impacts),
                        'lift_offs': len(lift_offs),
                        'cadence': len(groud_impacts) / duration,
                    })



        # Set the x-axis label for the last subplot
        axs[-1].set_xlabel("Time [s]")
        # Set the title for the entire figure
        fig.suptitle(f"{trial_name} - {"Axis: " + axis}", fontsize=14)
        # Adjust the layout to prevent overlap
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        cycles = {}
        for i in range(len(groud_impacts) - 1):
            for sensor in sensors.keys():
                cycle_df = sensors[sensor]['data'].copy()
                cycle_time_mask = (cycle_df.index >= groud_impacts[i][0]) & (cycle_df.index <= groud_impacts[i + 1][0])
                cycle_df = cycle_df[cycle_time_mask]
                # Normalize time to interval 0-1
                cycle_df.index = (cycle_df.index - groud_impacts[i][0]) / (groud_impacts[i + 1][0] - groud_impacts[i][0])
                if i not in cycles:
                    cycles[i] = {}
                cycles[i][sensor] = {
                    'data': cycle_df,
                    'trial_type': sensors[sensor]['trial_type'],
                    'side': sensors[sensor]['side'],
                    'label': sensors[sensor]['label'],
                    'color': sensors[sensor]['color'],

                }
        ranges = []

        for i in range(len(pushes) - 1):
            if '3' in sensors:
                cycle_df = sensors['3']['data'].copy()
                cycle_time_mask = (cycle_df.index >= pushes[i][0]) & (cycle_df.index <= pushes[i + 1][0])
                cycle_segment = cycle_df[cycle_time_mask]['Euler_Y']
                if not cycle_segment.empty:
                    max_val = cycle_segment.max()
                    min_val = cycle_segment.min()
                    ranges.append(max_val - min_val)

        # Pridaj do štatistiky nový stĺpec, ak máme rozsahy
        if ranges:
            stats[-1]['Euler_Y_range_avg'] = sum(ranges) / len(ranges)
        # Save the figure to the output directory
        output_path = output_dir / f"{trial_name.replace(' ', '_')}_{axis}.png"
        fig.savefig(output_path)
        plt.close(fig)

        if axis != 'Euler_Y':
            continue

        fig, axs = plt.subplots(4, 1, figsize=(25, 10), sharex=True)

        plot_variable(axs[0], '3', sensors['3'], 'Euler_Y')
        plot_variable(axs[1], '4', sensors['4'], 'FreeAcc_X')
        plot_variable(axs[2], '4', sensors['4'], 'FreeAcc_Z')
        plot_variable(axs[3], '5', sensors['5'], 'FreeAcc_X')
        #plot_variable(axs[3], '5', sensors['5'], 'Velocity_X')
        # Set the x-axis label for the last subplot
        axs[-1].set_xlabel("Time [s]")
        # Adjust the layout to prevent overlap
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for ax in axs:
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.grid(True)
            ax.grid(which='minor', linestyle=':', linewidth=0.5)

        # for ax in axs:
        #     ax.axvspan(pushes[0][0], lift_offs[0][0], color='green', alpha=0.1)
        #     ax.axvspan(lift_offs[0][0], pushes[1][0], color='red', alpha=0.1)

        # Save the figure to the output directory
        output_path = output_dir / f"{trial_name.replace(' ', '_')}_all.png"
        fig.savefig(output_path)
        plt.close(fig)

        plot_cycles(cycles)

        fig, axs = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

        for cycle in cycles.values():
            plot_variable(axs[0], '3', cycle['3'], 'Euler_Y', False)
            plot_variable(axs[1], '4', cycle['4'], 'FreeAcc_Z', False)
            plot_variable(axs[2], '5', cycle['5'], 'FreeAcc_X', False)
            plot_variable(axs[3], '5', cycle['5'], 'Velocity_X', False)
            # Set the x-axis label for the last subplot
            axs[-1].set_xlabel("Time [s]")
            # Adjust the layout to prevent overlap
            fig.tight_layout(rect=(0, 0, 1, 0.95))

        # for ax in axs:
        #     ax.axvspan(pushes[0][0], lift_offs[0][0], color='green', alpha=0.1)
        #     ax.axvspan(lift_offs[0][0], pushes[1][0], color='red', alpha=0.1)

        # Save the figure to the output directory
        output_path = output_dir / f"{trial_name.replace(' ', '_')}_cycles.png"
        fig.savefig(output_path)
        plt.close(fig)

# Získaj všetky unikátne kľúče zo štatistík
all_keys = set()
for row in stats:
    all_keys.update(row.keys())

# Usporiadaj ich
fieldnames = sorted(all_keys)

# Zaokrúhli číselné hodnoty na 4 desatinné miesta
rounded_stats = []
for row in stats:
    rounded_row = {}
    for key in fieldnames:
        value = row.get(key, "")
        if isinstance(value, float):
            rounded_row[key] = round(value, 4)
        else:
            rounded_row[key] = value
    rounded_stats.append(rounded_row)

# Zapíš CSV
with open("stats.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rounded_stats)
