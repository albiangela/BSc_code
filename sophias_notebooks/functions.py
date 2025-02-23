import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from scipy.signal import savgol_filter
import cv2
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.lines as mlines


    
# Define a function to load and preprocess data from npz files
def load_and_preprocess_outline_data(files,min_frame = np.inf ,max_frame=-np.inf,fs = 50):
    data = {}
    # min_frame = np.inf
    # max_frame = -np.inf
    screen = [np.inf, np.inf, -np.inf, -np.inf]
    #print(files)

    for f in files:
        #print("loading", f)
        data[f] = {}
        with np.load(f) as npz:
            midline = {}
            offset = npz["offset"]
            frames = npz["frames"]

            if min_frame > frames.min():
                min_frame = frames.min()
            if max_frame < frames.max():
                max_frame = frames.max()

            if offset.T[0].min() < screen[0]:
                screen[0] = offset.T[0].min()
            if offset.T[1].min() < screen[1]:
                screen[1] = offset.T[1].min()

            if offset.T[0].max() > screen[2]:
                screen[2] = offset.T[0].max()
            if offset.T[1].max() > screen[3]:
                screen[3] = offset.T[1].max()

            midline = {}
            if len(npz["midline_points"].shape) == 2:
                i = 0
                indices = []
                for l in npz["midline_lengths"][:-1]:
                    i += l
                    indices.append(int(i))
                points = np.split(npz["midline_points"], indices, axis=0)
                for frame, point, off in zip(frames, points, offset):
                    midline[frame] = point + off
            else:
                for mpt, off, frame in zip(npz["midline_points"], offset, frames):
                    midline[frame] = mpt + off

            i = 0
            indices = []
            for l in npz["outline_lengths"][:-1]:
                i += l
                indices.append(int(i))
            points = np.split(npz["outline_points"], indices, axis=0)
            outline = {}
            for frame, point, off in zip(frames, points, offset):
                outline[frame] = point + off

            # Handle holes
            hole_counts = npz["hole_counts"].astype(int)
            hole_points = npz["hole_points"]
            holes = {}
            count_index = 0
            point_index = 0

            for frame in frames:
                holes[frame] = []
                num_holes = hole_counts[count_index]
                count_index += 1
                
                for _ in range(num_holes):
                    num_points = hole_counts[count_index]
                    count_index += 1
                    
                    if point_index + num_points > len(hole_points):
                        raise Exception(f"Error: index {point_index + num_points} is out of bounds for hole_points with size {len(hole_points)}")
                        break
                    
                    #print(f"{frame}: {num_points} points at index {point_index}")
                    hole = hole_points[point_index:point_index + num_points]
                    holes[frame].append(hole)
                    point_index += num_points

            data[f]["holes"] = holes
            data[f]["midline"] = midline
            data[f]["outline"] = outline

    screen[0] -= 10
    screen[1] -= 10
    screen[2] *= 1.1
    screen[3] *= 1.1
    input_shape = (screen[2] - screen[0], screen[3] - screen[1])
    output_width = 1280
    output_shape = (output_width, int(output_width * input_shape[1] / input_shape[0]))  # Adjust output resolution to maintain aspect ratio

    return data, screen, input_shape, output_shape 

# Define a function to load and preprocess sharks data from npz files
def load_sharks_data(files):
    cm_per_pixel = 1
    sharks_data = []
    # files = sorted(glob.glob(os.path.join(folder_path, "*_sharks*.npz")))

    for f in files:
        with np.load(f) as npz:
            frames = npz["frame"]
            X = npz["X#pcentroid"] / cm_per_pixel
            Y = npz["Y#pcentroid"] / cm_per_pixel
            poseX = {key: npz[key] for key in npz if key.startswith("poseX")}
            poseY = {key: npz[key] for key in npz if key.startswith("poseY")}
            
            for i, frame in enumerate(frames):
                row = {"frame": frame, "X": X[i], "Y": Y[i], "ID": int(f.split("fish")[-1].split(".npz")[0])}
                for k, v in poseX.items():
                    row[k] = v[i]
                for k, v in poseY.items():
                    row[k] = v[i]
                sharks_data.append(row)

    sharks_df = pd.DataFrame(sharks_data)
    # print("Sharks DataFrame Head:\n", sharks_df.head())
    return sharks_df

def read_npz_files(folder_path, npz_files,base_filename, keys_to_extract, cm_per_pixel, moving_average_window, smooth=True):
    dataframes = []
    
    # Loop over each npz file
    for idx, filename in enumerate(np.sort(npz_files)):
        print(f"Processing file: {filename}") 
        filepath = os.path.join(folder_path, filename)
        
        # Load the npz file
        npz_data = np.load(filepath, allow_pickle=True)

        # Extract the required keys and frame data, ignoring the 2D key for now
        data = {}
        for key in keys_to_extract:
            if key in npz_data and npz_data[key].ndim == 1:
                data[key] = npz_data[key]

        # Extract pcentroid values and apply the cm_per_pixel conversion
        if "X#pcentroid" in npz_data and "Y#pcentroid" in npz_data:
            data["X"] = npz_data["X#pcentroid"] / cm_per_pixel
            data["Y"] = npz_data["Y#pcentroid"] / cm_per_pixel

        # Now convert the extracted 1D data into a DataFrame
        df = pd.DataFrame(data)
        df['frame'] = df['frame'].astype(int)

        # Skip files with fewer frames than the moving average window
        if df.shape[0] < moving_average_window:
            continue

        # Handle infinite values by replacing them with NaN
        df = transform_inf_to_nan(df)

        # Rename the pose columns to more descriptive names
        rename_mapping = {
            "poseX0": "headx", "poseY0": "heady", "poseX1": "lxfinx", "poseY1": "lxfiny",
            "poseX2": "btipx", "poseY2": "btipy", "poseX3": "rxfinx", "poseY3": "rxfiny",
            "poseX4": "pelvicfinx", "poseY4": "pelvicfiny", "poseX5": "sdfinx", "poseY5": "sdfiny",
            "poseX6": "pedunclex", "poseY6": "peduncley", "poseX7": "finx", "poseY7": "finy",
            "poseX8": "ttipx", "poseY8": "ttipy"
        }
        df.rename(columns=rename_mapping, inplace=True)

        # Now handle the 2D array for segment ranges (assuming the key is "your_key")
        if "frame_segments" in npz_data:  # Replace with the actual key name
            segment_data = npz_data["frame_segments"]  # This is the 2D array with start and end frames
            segment_col = np.zeros(df.shape[0], dtype=int)  # Initialize the segment column with zeros

            # Assign segment numbers based on start and end frame ranges
            for segment_num, (start, end) in enumerate(segment_data, start=1):
                segment_col[(df['frame'] >= start) & (df['frame'] <= end)] = segment_num

            # Add the segment column to the DataFrame
            df['segment'] = segment_col

        # Apply interpolation and smoothing (optional, commented out by default)
        # columns_to_process = df.columns.difference(['ID'])
        # for column in columns_to_process:
        #     df[column] = df[column].interpolate(method='linear')
        # if smooth:
        #     for column in columns_to_process:
        #         df[column] = savgol_filter(df[column], moving_average_window, polyorder=1)
        if smooth:
            columns_to_smooth = df.columns.difference(['ID', 'frame', 'segment'])
            for column in columns_to_smooth:
                df[column] = df[column].rolling(window=moving_average_window, min_periods=1).mean()

        # Extract fish ID from filename
        match = re.search(r"fish(\d+)", filename)
        if match:
            fid = int(match.group(1))
        else:
            fid = None

        # Insert the fish ID into the DataFrame
        df.insert(loc=1, column='ID', value=fid)

        # Append the DataFrame to the list of dataframes
        dataframes.append(df)

    # Concatenate all DataFrames into one
    processed_df = pd.concat(dataframes, ignore_index=True)
    processed_df['Trial'] = base_filename

    return processed_df


# Helper function to replace inf with NaN
def transform_inf_to_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)



def calculate_polygon_area(points):
    """Calculate the area of a polygon given its vertices using the shoelace formula."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))






def read_files(directory, extension, include_posture=False, prefix_filter=None):
    files_dict = {} 
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(f'.{extension}') and (include_posture or 'posture' not in file):
                if prefix_filter is None or file.startswith(prefix_filter):
                    files_dict.setdefault(root, []).append(file)
    return files_dict


def get_trex_settings_value(settings_file_path,variable):
    var = None
    
    with open(settings_file_path, 'r') as file:
        for line in file:
            if variable in line:
                # Assuming the line format is "cm_per_pixel: value"
                key, value = line.strip().split('=')
                if key.strip() == variable:
                    var = float(value.strip())
                    break

    return var


# Function to find consecutive frames without NaN
def find_consecutive_non_nan_section(df, column_names, length):
    valid_frames = df[column_names].notna().all(axis=1)
    max_count = 0
    max_start = -1
    current_count = 0
    current_start = 0

    for i, is_valid in enumerate(valid_frames):
        if is_valid:
            if current_count == 0:
                current_start = i
            current_count += 1
        else:
            if current_count >= length and current_count > max_count:
                max_count = current_count
                max_start = current_start
            current_count = 0

    # Check the last segment
    if current_count >= length and current_count > max_count:
        max_count = current_count
        max_start = current_start

    if max_start != -1:
        return df.iloc[max_start:max_start + length]
    else:
        return None
        
def load_files(directory, extension, include_posture=False, prefix_filter=None):
    files_dict = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(f'.{extension}') and (include_posture or 'posture' not in file):
                if prefix_filter is None or file.startswith(prefix_filter):
                    files_dict.setdefault(root, []).append(file)
    return files_dict

files_dict = {0,1}
def filter_files(files_dict, videofile_prefix):
    # Construct the pattern dynamically based on the videofile_prefix
    pattern = re.compile(rf'{re.escape(videofile_prefix)}_fish\d\.npz')
    return [file for files in files_dict.values() for file in files if pattern.match(file)]



def extract_trial_name(filename):
    """
    Extracts the trial name from a given filename.

    The filename is expected to have its components separated by underscores (_). 
    The trial name is assumed to be the second component of the filename.

    Parameters:
    - filename (str): The input filename string.

    Returns:
    - str: The extracted trial name.
    """
    return filename.split('_')[1]

def load_files(directory, filetype, include_posture, prefix_filter=None):
    fish_data = {}
    
    for filename in np.sort(os.listdir(directory)):
        # Check if the filename starts with the prefix_filter if it's provided
        if prefix_filter and not filename.startswith(prefix_filter):
            continue  # Skip the file if it doesn't match the prefix_filter

        if filename.endswith(filetype):
            # Check if '_posture_' is in the filename
            if ('_posture_' in filename) != include_posture:
                continue  # Skip the file if condition not met

            trial_name = extract_trial_name(filename)
            # print(trial_name)
            if trial_name not in fish_data:
                fish_data[trial_name] = []
            fish_data[trial_name].append(filename)
    return fish_data

# # Function to calculate angles between consecutive points in the skeleton
# def skeletons_to_angles(skeleton):
#     '''skeleton: array(nframes, 2)'''
#     skelX = skeleton[:, 0]
#     skelY = skeleton[:, 1]
#     dX = np.diff(skelX, axis=0)
#     dY = np.diff(skelY, axis=0)
    
#     # Calculate tangent angles. atan2 uses angles from -pi to pi
#     angles = np.arctan2(dY, dX)
#     angles = np.mod(angles,2*np.pi)
#     return angles

# # Function to calculate angles and add them to a DataFrame for each frame
# def calculate_angles_for_df(df, midpoints_centered):
#     """
#     Calculate angles for the rotated skeleton points in the DataFrame.

#     Parameters:
#     df (DataFrame): A pandas DataFrame containing rotated skeleton points.
#     midpoints_centered (list): A list of column name pairs representing x and y coordinates of skeleton points.

#     Returns:
#     DataFrame: The original DataFrame with additional columns for angles between joints.
#     """
#     # Prepare for storing angles and maintaining ID, segment, and frame
#     angles_list = []
#     ids_segments_frames = []

#     # Loop over each frame in the dataframe
#     for idx, row in df.iterrows():
#         # Extract the rotated x and y points for the current frame
#         skeleton = np.array([[row[x_col], row[y_col]] for (x_col, y_col) in midpoints_centered])
#         # Calculate angles for the current skeleton
#         angles = skeletons_to_angles(skeleton)
        
#         # Store the angles and corresponding metadata (ID, segment, frame)
#         angles_list.append(angles)
#         ids_segments_frames.append((row['ID'], row['segment'], row['frame']))
    
#     # Convert list of angles to a DataFrame, including ID, segment, and frame for merging later
#     angles_df = pd.DataFrame(
#         angles_list,
#         columns=[f"angle_joint_{i}" for i in range(len(midpoints_centered) - 1)]
#     )
    
#     # Add the ID, segment, and frame columns to the angles DataFrame for merging
#     angles_df[['ID', 'segment', 'frame']] = pd.DataFrame(ids_segments_frames, columns=['ID', 'segment', 'frame'])

#     # Return the new angles dataframe
#     return angles_df

# # Apply the function to each group and merge it back with the original dataframe
# def process_angles(data2, midpoints):
#     # Apply the calculation to each group and concatenate the results
#     angles_df = data2.groupby(['ID', 'segment']).apply(calculate_angles_for_df, midpoints)
#     angles_df = angles_df.reset_index(drop=True)  # Reset index for easier merging

#     # Merge the angles back with the original dataframe on ID, segment, and frame
#     df_merged = pd.merge(data2, angles_df, on=['ID', 'segment', 'frame'], how='left')
    
#     return df_merged


# def load_files(directory, filetype, include_posture):
#     fish_data = {}
    
#     for filename in np.sort(os.listdir(directory)):
#         if filename.endswith(filetype):
#             # Check if '_posture_' is in the filename
#             if ('_posture_' in filename) != include_posture:
#                 continue  # Skip the file if condition not met

#             trial_name = extract_trial_name(filename)
#             # print(trial_name)
#             if trial_name not in fish_data:
#                 fish_data[trial_name] = []
#             fish_data[trial_name].append(filename)
#     return fish_data

# Helper function to extract trial name (assumed to be present in your existing code)
# def extract_trial_name(filename):
#     # Implement the actual extraction logic based on your filename format
#     # Placeholder implementation: just return the filename without extension
#     return os.path.splitext(filename)[0]

# moving_average_window = 50

# Function to handle mouse clicks and store the coordinates
def mouse_click(event, x, y, flags, param):
    global df
    # Check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the point to the dataframe if less than 3 points are recorded
        if len(df) < 3:
            df.loc[len(df)] = [x, y]
            # If 3 points are clicked, set the flag to True
            if len(df) == 3:
                param[0] = True




def interpolate_nans(df):
    return df.interpolate(limit=50)

def calculate_speed_acceleration(df, colx,coly,fps, step_size, max_speed=500000, max_acceleration=500000):
    df['delta_time'] = df['frame'].diff() / fps
    df['delta_x'] = df[colx].diff()
    df['delta_y'] = df[coly].diff()
    
    df['speed'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2) / df['delta_time']
    df['acceleration'] = df['speed'].diff() / (df['delta_time'] * step_size)
    
    # Apply filters to avoid high speed and acceleration values
    df.loc[df['speed'] > max_speed, 'speed'] = np.nan
    df.loc[df['acceleration'] > max_acceleration, 'acceleration'] = np.nan
    
    df.drop(columns=['delta_x', 'delta_y'], inplace=True)
    return df



def pairwise_distance(ffx_centroid, ffy_centroid, nfx_centroid, nfy_centroid, numfish):
    '''non-vectorized code for pairwise distance'''
    distance = np.sqrt((nfx_centroid-ffx_centroid)**2 + (nfy_centroid-ffy_centroid)**2)
    distance[distance == np.inf] = np.nan  
    return distance 

def get_nndist(df):

    """With this fuction I calculate:
    - nearest neighbor distance
    - relative heading with nearest neighbour"""

    fishlist = np.unique(df["ID"].values)
    maxframe = len(np.unique(df["frame"].values))
    num_individuals = len(np.unique(df["ID"].values))
    pairwise_df = np.zeros((num_individuals, num_individuals, maxframe))
    sel_fish = df.groupby("ID")

    #     df_pairwise = []
    for ixff, fish in enumerate(fishlist):

        ff_ = sel_fish.get_group(fish)
        x_ff_ = ff_["x"].values
        x_ff_ = np.asarray(list(x_ff_))

        y_ff_ = ff_["y"].values
        y_ff_ = np.asarray(list(y_ff_))

        for ixnf, nf in enumerate(fishlist):  # list_no_fish:

            nf_ = sel_fish.get_group(nf)
            x_nf_ = nf_["x"].values
            x_nf_ = np.asarray(list(x_nf_))

            y_nf_ = nf_["y"].values
            y_nf_ = np.asarray(list(y_nf_))

            pairwise_df[ixff, ixnf, :] = pairwise_distance(x_ff_, y_ff_, x_nf_, y_nf_,num_individuals)


    lst = np.arange(num_individuals)

    distinf = pairwise_df.T
    headinf = pairwise_heading.T
    
    distinf[:, lst, lst] = np.nan
    headinf[:, lst, lst] = np.nan

    mindist = np.nanmin(distinf, axis=1)


    return mindist

def calculate_direction_vector(df,colx,coly,step):

    df['dir_x'] = df[colx].diff(periods = step) #np.cos(df['heading'])
    df['dir_y'] = df[coly].diff(periods = step)
    df['heading'] = np.arctan2(df['dir_y'],df['dir_x'])
    
    return df


def calculate_angular_velocity(df, angle_column, time_step,steps):
    """
    Calculate the angular velocity.

    :param df: pandas DataFrame containing the data.
    :param angle_column: string, name of the column in df that contains the angles.
    :param time_step: float, the time interval between each measurement.
    :return: pandas DataFrame, the input dataframe with an additional column 'angular_velocity' containing the calculated angular velocities.
    """
    angle_diff = df[angle_column].diff(periods = steps)
    
    # Handle the wrap-around for angles
    angle_diff[angle_diff > np.pi] -= 2 * np.pi
    angle_diff[angle_diff < -np.pi] += 2 * np.pi
    
    # Calculate angular velocity
    df['angular_velocity'] = angle_diff / time_step
    
    # Handle the first row
    df.iloc[0, df.columns.get_loc('angular_velocity')] = 0
    
    return df

# def read_npz_files(folder_path, npz_files, keys_to_extract, cm_per_pixel, original_video_height, moving_average_window, smooth=True):
#     # h_in_cm = original_video_height * cm_per_pixel

#     dataframes = []
#     for idx, filename in enumerate(np.sort(npz_files)):
#         print(filename) 
#         filepath = os.path.join(folder_path, filename)
#         npz_data = np.load(filepath, allow_pickle=True)
#         data = {key: npz_data[key] for key in keys_to_extract}
#         df = pd.DataFrame(data)
#         df['frame'] = df['frame'].astype(int)

#         if df.shape[0] < moving_average_window:
#             continue

#         df = transform_inf_to_nan(df)
#         # df = interpolate_nans(df)

#         # Column renaming
#         rename_mapping = {
#             "poseX0": "headx", "poseY0": "heady", "poseX1": "lxfinx", "poseY1": "lxfiny",
#             "poseX2": "btipx", "poseY2": "btipy", "poseX3": "rxfinx", "poseY3": "rxfiny",
#             "poseX4": "pelvicfinx", "poseY4": "pelvicfiny", "poseX5": "sdfinx", "poseY5": "sdfiny",
#             "poseX6": "pedunclex", "poseY6": "peduncley","poseX7": "finx", "poseY7": "finy",
#             "poseX8": "ttipx", "poseY8": "ttipy"
#         }
#         df.rename(columns=rename_mapping, inplace=True)

        
#         # Apply interpolation and smoothing to all columns except 'ID'
#         columns_to_process = df.columns.difference(['ID'])
#         # for column in columns_to_process:
#             # df[column] = df[column].interpolate(method='linear')
#         # if smooth:
#         #     df[column] = savgol_filter(df[column], moving_average_window, polyorder=1)

#         match = re.search(r"fish(\d+)", filename)
#         if match:
#             fid = int(match.group(1))
#         else:
#             fid = None

#         df.insert(loc=1, column='ID', value=fid)
#         dataframes.append(df)

#     processed_df = pd.concat(dataframes, ignore_index=True)

#     return processed_df
    
# def read_npz_files(folder_path, npz_files, keys_to_extract, cm_per_pixel, original_video_height, moving_average_window, fps, step_size, step_size_angle, smooth=True):
#     h_in_cm = original_video_height * cm_per_pixel

    
#     dataframes = []
#     for idx, filename in enumerate(np.sort(npz_files)):
#         print(filename)
#         filepath = os.path.join(folder_path, filename)
#         npz_data = np.load(filepath, allow_pickle=True)

#         data = {key: npz_data[key] for key in keys_to_extract}
#         df = pd.DataFrame(data)
#         frame_data = data['frame']
#         # print(df)
#         # df['frame'] = frame_data.astype(int)
#         # df['frame'] = df['frame'].astype(int)

#         if df.shape[0] < moving_average_window:
#             continue

#         df = transform_inf_to_nan(df)
#         df = interpolate_nans(df)

#         # Column renaming
#         rename_mapping = {
#             "poseX0": "headx", "poseY0": "heady", "poseX1": "rxfinx", "poseY1": "rxfiny",
#             "poseX2": "btipx", "poseY2": "btipy", "poseX3": "lxfinx", "poseY3": "lxfiny",
#             "poseX4": "pfinx", "poseY4": "pfiny", "poseX5": "sdfinx", "poseY5": "sdfiny",
#             "poseX6": "pedx", "poseY6": "pedy","poseX7": "finx", "poseY7": "finy","poseX8": "ttipx","poseY8":"ttipy"
#         }
#         df.rename(columns=rename_mapping, inplace=True)
#         # Multiply all body columns by cm_per_pixel to convert to centimeters
#         body_columns = ["headx","heady", "rxfinx", "rxfiny","btipx", "btipy", "lxfinx", "lxfiny",
#                         "pfinx", "pfiny","sdfinx", "sdfiny", "pedx", "pedy", "finx", "finy","ttipx","ttipy"]

#         for column in body_columns:
#             if column in df.columns:
#                 df[column] = df[column] * cm_per_pixel
        
#         # Apply interpolation and smoothing to all columns except 'ID'
#         columns_to_process = df.columns.difference(['ID','frame'])
#         for column in columns_to_process:
#             df[column] = df[column].interpolate(method='linear')
#             if smooth:
#                 try:
#                     df[column] = savgol_filter(df[column], moving_average_window, polyorder=1)
#                 except LinAlgError:
#                     print(f"LinAlgError: SVD did not converge for column {column}. Skipping smoothing for this column.")
#                     continue
#                 # df[column] = savgol_filter(df[column], moving_average_window, polyorder=1)
#         # print(step_size_angle)
#         df = calculate_direction_vector(df,  'headx', 'heady',step=step_size_angle)
#         df = calculate_speed_acceleration(df, 'headx', 'heady', fps=fps, step_size=step_size)
#         df = calculate_angular_velocity(df, 'heading', time_step=1/fps, steps=4)

#         match = re.search("fish(\d+)", filename)
#         if match:
#             fid = int(match.group(1))
#         else:
#             fid = None

#         df.insert(loc=1, column='ID', value=fid)
#         dataframes.append(df)

#     processed_df = pd.concat(dataframes, ignore_index=True)

#     return processed_df


# def calculate_direction_vector(df,colx,coly,step):

#     df['dir_x'] = df[colx].diff(periods = step) #np.cos(df['heading'])
#     df['dir_y'] = df[coly].diff(periods = step)
#     df['heading'] = np.arctan2(df['dir_y'],df['dir_x'])
    
#     return df

def calculate_heading(df):
    """
    Calculate the heading angle based on headx, heady and btipx, btipy.

    Heading is calculated as the arctangent of the slope formed by the line
    between the points (headx, heady) and (btipx, btipy).

    Parameters:
    df (DataFrame): A pandas DataFrame that contains columns headx, heady, btipx, and btipy.

    Returns:
    DataFrame: The original DataFrame with a new 'heading' column.
    """
    # Calculate the differences in x and y coordinates
    dx = df['btipx'] - df['headx']
    dy = df['btipy'] - df['heady']
    
    # Compute the heading angle in radians
    heading_radians = np.arctan2(dy, dx)
    
    # Convert heading to degrees for better interpretability
    heading_degrees = np.degrees(heading_radians)
    
    # Normalize the heading to a range of [0, 360)
    heading_degrees = (heading_degrees + 360) % 360
    
    # Add the heading column to the DataFrame
    df['heading'] = heading_degrees
    
    return df
    



def get_nndist_heading(df):

    """With this fuction I calculate:
    - nearest neighbor distance
    - relative heading with nearest neighbour"""

    fishlist = np.unique(df["ID"].values)
    maxframe = len(np.unique(df["frame"].values))
    numfish = len(np.unique(df["ID"].values))
    pairwise_df = np.zeros((numfish, numfish, maxframe))
    pairwise_heading = np.zeros((numfish, numfish, maxframe))
    pairwise_polarization = np.zeros((numfish, numfish, maxframe))
    sel_fish = df.groupby("ID")

    #     df_pairwise = []
    for ixff, fish in enumerate(fishlist):

        # list_all = [0,1,2,3,4,5,6,7]
        # list_no_fish = [x for x in list_all if x != fish]
        #     print(list_no_fish)
        ff_ = sel_fish.get_group(fish)

        x_ff_ = ff_["x"].values
        x_ff_ = np.asarray(list(x_ff_))

        y_ff_ = ff_["y"].values
        y_ff_ = np.asarray(list(y_ff_))

        dirxff_ = ff_["dir_x"].values
        dirxff_ = np.asarray(list(dirxff_))

        diryff_ = ff_["dir_y"].values
        diryff_ = np.asarray(list(diryff_))

        # psi_ff_ = ff_["heading"].values
        # psi_ff_ = np.asarray(list(psi_ff_))

        #         nf_idx = 0
        for ixnf, nf in enumerate(fishlist):  # list_no_fish:

            #         print(fish,nf)
            #             print(fish,nf)
            nf_ = sel_fish.get_group(nf)

            x_nf_ = nf_["x"].values
            x_nf_ = np.asarray(list(x_nf_))

            y_nf_ = nf_["y"].values
            y_nf_ = np.asarray(list(y_nf_))

            dirxnf_ = nf_["dir_x"].values
            dirxnf_ = np.asarray(list(dirxnf_))

            dirynf_ = nf_["dir_y"].values
            dirynf_ = np.asarray(list(dirynf_))

            psi_nf_ = nf_["heading"].values
            psi_nf_ = np.asarray(list(psi_nf_))

            #             pdist = pairwise_distance(x_ff_,y_ff_,x_nf_,y_nf_)
            #         print(test.shape)
            pairwise_df[ixff, ixnf, :] = pairwise_distance(x_ff_, y_ff_, x_nf_, y_nf_,numfish = 4)

            # pairwise_heading[ixff, ixnf, :] = get_relheading(np.array([dirxff_, diryff_]), np.array([dirxnf_, dirynf_]))

    lst = np.arange(numfish)

    distinf = pairwise_df.T
    # headinf = pairwise_heading.T
    
    distinf[:, lst, lst] = np.nan
    # headinf[:, lst, lst] = np.nan

    mindist = np.nanmin(distinf, axis=1)

    # if len(fishlist) == 1:
    #     nnheading = np.repeat(np.inf, distinf.shape[0])

    # """calculate nearest neighbour heading"""
    # nnheading = np.zeros((distinf.shape[0], distinf.shape[1]))
    # for f in range(distinf.shape[0]):
    #     try:
    #         nnheading[f, :] = np.nanargmin(distinf[f, :, :], axis=1).choose(
    #             headinf[f, :, :]
    #         )
    #     except ValueError:
    #         nnheading[f, :] = np.nan

    return mindist #, nnheading

# def process_fish_data(directory, fish_data, fps=25, step_size=1):
#     processed_dfs = []
#     for trial_name, files in fish_data.items():
#         for filename in np.sort(files):
#             npz_data = np.load(os.path.join(directory, filename))
#             df = pd.DataFrame({key: npz_data[key] for key in ['frame', 'X#wcentroid','Y#wcentroid']})
#             df = transform_inf_to_nan(df)
#             df = interpolate_nans(df)
#             df = calculate_speed_acceleration(df, fps=fps, step_size=1)
#             df['Trial'] = trial_name
#             # df['FishID'] = int(re.search(r'fish(\d+)', filename).group(1))
#             # fid = int(filename.split('_')[-1].split('.')[0][-1:])
#             match = re.search("fish(\d+)", filename)
#             if match:
#                 fid = int(match.group(1))
#             else:
#                 fid = None  # or some default value
                
#             # print(fid)
#             df.insert(loc=1,column='FishID',value = fid)
#             df['FishID'] = df['FishID'].astype(int)
#             # df = df[(df['frame'] > minframe) & (df['frame'] < maxframe)]
#             df.rename(columns={"X#wcentroid":"x","Y#wcentroid":"y"},inplace = True)

#             processed_dfs.append(df)

#     processed_df = pd.concat(processed_dfs)
#     # Calculate nearest neighbor distance for each fish
#     total_frames = processed_df['frame'].nunique()
#     # nearest_neighbor_distances = calculate_nearest_neighbor_distance(processed_df, total_frames, num_individuals = 6)

#     # nndist = get_nndist(processed_df)

#     # # Add nearest neighbor distances to the corresponding 'FishID' group
#     # for idf, fish in enumerate(np.sort(processed_df.FishID.unique())):
#     #     processed_df.loc[processed_df['FishID'] == fish, 'nn_dist'] = nndist[:,idf]
#     return processed_df
    
    

### Plotting 
def plot_random_frames(df, num_frames_to_plot):
    # Get a random sample of frames
    random_frame = np.random.choice(df['frame'].unique()-num_frames_to_plot)
    random_frames = np.arange(random_frame,random_frame+num_frames_to_plot)

    # Create a colormap for differentiating fish trajectories
    num_fish = df['ID'].nunique()
    cmap = plt.cm.get_cmap('viridis', num_fish)

    # Plot X and Y positions for each fish trajectory
    for fish_id, group in df.groupby('ID'):
        plt.plot(group.loc[group['frame'].isin(random_frames), 'x'],
                    group.loc[group['frame'].isin(random_frames), 'y'],
                    label=f"Fish {fish_id}", alpha=0.8, c=cmap(fish_id))

    plt.xlabel('X [cm]')
    plt.ylabel('Y [cm]')
    plt.legend(title='Fish ID', loc='upper right')
    plt.title('Random Frame range Trajectories')
    # plt.grid(True)
    plt.show()



