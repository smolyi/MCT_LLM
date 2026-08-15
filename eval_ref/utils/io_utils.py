import os
import re
import argparse
import shutil
import json
import csv
import numpy as np
import pandas as pd
from typing import List, Set, Any


class ValidateFile(argparse.Action):
    """
    Module to validate files
    """
    def __call__(self, parser, namespace, values, option_string = None):
        
        if not os.path.exists(values):
            parser.error(f"Please enter a valid file path. Got: {values}")
        elif not os.access(values, os.R_OK):
            parser.error(f"File {values} doesn't have read access")
        setattr(namespace, self.dest, values)


def validate_file_path(input_string: str) -> str:
    """
    Validates whether the input string matches a file path pattern

    :param str input_string: input string
    :return: validated file path
    :rtype: str
    ::

        file_path = validate_file_path(input_string)
    """
    file_path_pattern = r"^[a-zA-Z0-9_\-\/.#]+$"

    if re.match(file_path_pattern, input_string):
        return input_string
    else:
        raise ValueError(f"Invalid file path: {input_string}")


def load_csv_to_dataframe_from_file(file_path: str, column_names: List[str], camera_ids: Set, interval: int = 1) -> pd.DataFrame:
    """
    Loads dataframe from a CSV file

    :param str file_path: file path
    :param List[str] column_names: column names
    :return: dataframe in the file
    :rtype: pd.DataFrame
    ::

        dataFrame = load_csv_to_dataframe_from_file(file_path, column_names)
    """
    data: List[List[str]] = list()
    valid_file_path = validate_file_path(file_path)
    
    df = pd.read_csv(valid_file_path, sep=" ", header=None, names=column_names, dtype={"CameraId": int, "Id": int, "FrameId": int})
    
    # Ensure non-negative values for CameraId, Id, FrameId
    if (df[['CameraId', 'Id', 'FrameId']] < 0).any().any():
        raise ValueError("Invalid negative values found for CameraId, Id, or FrameId.")


    # Filter by camera_id
    df = df[df['CameraId'].isin(camera_ids)]
    
    # Filter rows where FrameId % interval == 0
    df = df[df['FrameId'] % interval == 0]
    
    # Round the last two columns (assuming these are 'Xworld' and 'Yworld')
    df['Xworld'] = df['Xworld'].round(3)
    df['Yworld'] = df['Yworld'].round(3)
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty after filtering process.")

    return df


def write_dataframe_to_csv_to_file(file_path: str, data: pd.DataFrame, delimiter: str = " ") -> None:
    """
    Writes dataframe to a CSV file

    :param str file_path: file path
    :param pd.DataFrame data: dataframe to be written
    :param str delimiter: delimiter of the CSV file
    :return: None
    ::

        write_dataframe_to_csv_to_file(file_path, data, delimiter)
    """
    data.to_csv(file_path, sep=delimiter, index=False, header=False)


def make_dir(dir_path: str) -> None:
    """
    Makes a directory without removing other files

    :param str dir_path: directory path
    :return: None
    ::

        make_dir(dir_path)
    """
    valid_dir_path = validate_file_path(dir_path)
    if not os.path.isdir(valid_dir_path):
        os.makedirs(validate_file_path(dir_path))


def make_seq_maps_file(file_dir: str, scenes: List[str], benchmark: str, split_to_eval: str) -> None:
    """
    Makes a sequence-maps file used by TrackEval library

    :param str file_dir: file path
    :param Set(str) sensor_ids: names of sensors
    :param str benchmark: name of the benchmark
    :param str split_to_eval: name of the split of data
    :return: None
    ::

        make_seq_maps_file(file_dir, sensor_ids)
    """
    make_clean_dir(file_dir)
    file_name = benchmark + "-" +split_to_eval + ".txt"
    seq_maps_file = file_dir +  "/" + file_name
    f = open(seq_maps_file, "w")
    f.write("name\n")
    for name in scenes:
        sensor_name = str(name) + "\n"
        f.write(sensor_name)
    # f.write("FINAL")
    f.close()


def make_seq_ini_file(gt_dir: str, scene: str, seq_length: int) -> None:
    """
    Makes a sequence-ini file used by TrackEval library

    :param str gt_dir: file path
    :param str scene: Name of a single scene 
    :param int seq_length: Number of frames

    :return: None
    ::

        make_seq_ini_file(gt_dir, scene, seq_length)
    """
    ini_file_name = gt_dir + "/seqinfo.ini"
    f = open(ini_file_name, "w")
    f.write("[Sequence]\n")
    name= "name=" +str(scene)+ "\n"
    f.write(name)
    f.write("imDir=img1\n")
    f.write("frameRate=30\n")
    seq = "seqLength=" + str(seq_length) + "\n"
    f.write(seq)
    f.write("imWidth=1920\n")
    f.write("imHeight=1080\n")
    f.write("imExt=.jpg\n")
    f.close()


def get_scene_to_camera_id_dict(file_path):
    """
    Loads a mapping of scene names to camera IDs from a JSON file.

    :param str file_path: Path to the JSON file containing scenes data.
    :return: A dictionary where keys are scene names and values are lists of camera IDs.
    ::

        scene_to_camera_id_dict = get_scene_to_camera_id_dict(file_path)
    """
    scene_2_cam_id = dict()
    valid_file_path = validate_file_path(file_path)
    with open(valid_file_path, "r") as file:
        scenes_data = json.load(file)
        for scene_data in scenes_data:
            scene_name = scene_data["scene_name"]
            camera_ids = scene_data["camera_ids"]
            if scene_name not in scene_2_cam_id:
                scene_2_cam_id[scene_name] = []
            scene_2_cam_id[scene_name].extend(camera_ids)
    return scene_2_cam_id
    

def check_file_size(file_path):
    """
    Checks the size of a file and raises an exception if it exceeds 2 GB.

    :param str file_path: Path to the file to be checked.
    :return: None
    :raises ValueError: If the file size is greater than 2 GB.
    ::

        check_file_size(file_path)
    """
    file_size_bytes = os.path.getsize(file_path)
    file_size_gb = file_size_bytes / (2**30)
    if file_size_gb > 2:
        raise ValueError(f"The size of the file is {file_size_gb:.2f} GB, which is greater than the 2 GB")


def make_clean_dir(dir_path: str) -> None:
    """
    Makes a clean directory
    :param str dir_path: directory path
    :return: None
    ::
        make_clean_dir(dir_path)
    """
    valid_dir_path = validate_file_path(dir_path)
    if os.path.exists(valid_dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    if not os.path.isdir(valid_dir_path):
        os.makedirs(validate_file_path(dir_path))
