#!/usr/bin/python3
"""
Evaluation script for the Multi-Camera People Tracking track in the AI City Challenge, Track 1 in 2024.

# Environment setup:
[Optional] conda create -n aicity24-track1 python=3.10
[Optional] conda activate aicity24-track1

pip3 install pandas
pip3 install matplotlib
pip3 install scipy

# Usage: Set number of cores based on your cpu core count.

python3 aicityeval-track1.py --prediction_file ./sample_file/pred.txt --ground_truth_file ./sample_file/ground_truth_test_full.txt --num_cores 16 --scene_2_camera_id_file ./sample_file/scene_name_2_cam_id_full.json

python3 aicityeval-track1.py --prediction_file ./sample_file/pred.txt --ground_truth_file ./sample_file/ground_truth_test_half.txt --num_cores 16 --scene_2_camera_id_file ./sample_file/scene_name_2_cam_id_half.json


"""
import os
import sys
import time
import tempfile
import trackeval
import pandas as pd
import numpy as np


from argparse import ArgumentParser, ArgumentTypeError
from typing import List
from utils.io_utils import load_csv_to_dataframe_from_file, write_dataframe_to_csv_to_file, make_seq_maps_file, make_seq_ini_file, make_dir, check_file_size, get_scene_to_camera_id_dict


def get_unique_entry_per_scene(dataframe, scene_name, scene_2_camera_id):
    camera_ids = scene_2_camera_id[scene_name]
    filtered_df = dataframe[dataframe["CameraId"].isin(camera_ids)]
    unique_entries_df = filtered_df.drop_duplicates(subset=["FrameId", "Id"])
    return unique_entries_df

def check_positive(value):
    int_value = int(value)
    if int_value <= 0:
        raise ArgumentTypeError(f"{value} is an invalid num of cores")
    return int_value

def computes_mot_metrics(prediction_file_path: str, ground_truth_path: str, output_dir: str, num_cores: int, scene_2_cam_id_file: str) -> None:

    
    check_file_size(prediction_file_path)

    # Create a temp directory if output_dir is not specified
    is_temp_dir = False
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        is_temp_dir = True
        output_dir = temp_dir.name
        print(f"Temp files will be created here: {output_dir}")

    # Create a scene 2 camera_id dict
    scene_2_camera_id = get_scene_to_camera_id_dict(scene_2_cam_id_file)
    camera_ids = {camera_id for camera_ids in scene_2_camera_id.values() for camera_id in camera_ids}

    # Load ground truth and prediction files in dataframe
    column_names = ["CameraId", "Id", "FrameId", "X", "Y", "Width", "Height", "Xworld", "Yworld"]
    mot_pred_dataframe = load_csv_to_dataframe_from_file(prediction_file_path, column_names, camera_ids)
    ground_truth_dataframe = load_csv_to_dataframe_from_file(ground_truth_path, column_names, camera_ids)


    # Create evaluater configs for trackeval lib
    default_eval_config = trackeval.eval.Evaluator.get_default_eval_config()
    default_eval_config["PRINT_CONFIG"] = False
    default_eval_config["USE_PARALLEL"] = True
    default_eval_config["LOG_ON_ERROR"] = None
    default_eval_config["NUM_PARALLEL_CORES"] = num_cores

    # Create dataset configs for trackeval lib
    default_dataset_config = trackeval.datasets.MotChallenge3DLocation.get_default_dataset_config()
    default_dataset_config["DO_PREPROC"] = False
    default_dataset_config["SPLIT_TO_EVAL"] = "all"
    default_dataset_config["GT_FOLDER"] = os.path.join(output_dir, "evaluation", "gt")
    default_dataset_config["TRACKERS_FOLDER"] = os.path.join(output_dir, "evaluation", "scores")
    default_dataset_config["PRINT_CONFIG"] = False

    # Make output directory for storing results
    make_dir(default_dataset_config["GT_FOLDER"])
    make_dir(default_dataset_config["TRACKERS_FOLDER"])

    # Create sequence maps file for evaluation
    seq_maps_file = os.path.join(default_dataset_config["GT_FOLDER"], "seqmaps")
    make_seq_maps_file(seq_maps_file, scene_2_camera_id.keys(), default_dataset_config["BENCHMARK"], default_dataset_config["SPLIT_TO_EVAL"])

    # Set the metrics to obtain
    default_metrics_config = {"METRICS": ["HOTA"], "THRESHOLD": 0.5}
    default_metrics_config["PRINT_CONFIG"] = False
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}


    # Create prediction and ground truth list
    for scene_name in scene_2_camera_id.keys():
        
        # Convert ground truth multi-camera dataframe to single camera in MOT format
        ground_truth_dataframe_per_scene = get_unique_entry_per_scene(ground_truth_dataframe, scene_name, scene_2_camera_id)
        ground_truth_dataframe_per_scene = ground_truth_dataframe_per_scene[["FrameId", "Id", "X", "Y", "Width", "Height", "Xworld", "Yworld"]]
        ground_truth_dataframe_per_scene = ground_truth_dataframe_per_scene.sort_values(by="FrameId")

        # Make ground truth frame-ids 1-based
        ground_truth_dataframe_per_scene["FrameId"] += 1

        # Set other defaults
        ground_truth_dataframe_per_scene["Conf"] = 1
        ground_truth_dataframe_per_scene["Zworld"] = -1
        ground_truth_dataframe_per_scene = ground_truth_dataframe_per_scene[["FrameId", "Id", "X", "Y", "Width", "Height", "Conf", "Xworld", "Yworld", "Zworld"]]

        # Remove logs for negative frame ids
        ground_truth_dataframe_per_scene = ground_truth_dataframe_per_scene[ground_truth_dataframe_per_scene["FrameId"] >= 1]

        # Save single camera ground truth in MOT format as CSV
        mot_version = default_dataset_config["BENCHMARK"] + "-" + default_dataset_config["SPLIT_TO_EVAL"]
        gt_dir = os.path.join(default_dataset_config["GT_FOLDER"], mot_version)
        dir_name = os.path.join(gt_dir, str(scene_name))
        gt_file_dir = os.path.join(gt_dir, str(scene_name), "gt")
        gt_file_name = os.path.join(gt_file_dir, "gt.txt")
        make_dir(gt_file_dir)
        write_dataframe_to_csv_to_file(gt_file_name, ground_truth_dataframe_per_scene)

        # Convert predicted multi-camera dataframe to MOT format
        mot_pred_dataframe_per_scene = get_unique_entry_per_scene(mot_pred_dataframe, scene_name, scene_2_camera_id)
        mot_pred_dataframe_per_scene = mot_pred_dataframe_per_scene[["FrameId", "Id", "X", "Y", "Width", "Height", "Xworld", "Yworld"]]
        mot_pred_dataframe_per_scene = mot_pred_dataframe_per_scene.sort_values(by="FrameId")

        # Make MOT prediction frame-ids 1-based
        mot_pred_dataframe_per_scene["FrameId"] += 1

        # Remove logs for negative frame ids
        mot_pred_dataframe_per_scene = mot_pred_dataframe_per_scene[mot_pred_dataframe_per_scene["FrameId"] >= 1]

        # Set other defaults
        mot_pred_dataframe_per_scene["Conf"] = 1
        mot_pred_dataframe_per_scene["Zworld"] = -1
        mot_pred_dataframe_per_scene = mot_pred_dataframe_per_scene[["FrameId", "Id", "X", "Y", "Width", "Height", "Conf", "Xworld", "Yworld", "Zworld"]]
        
        # Save single camera prediction in MOT format as CSV
        mot_file_dir = os.path.join(default_dataset_config["TRACKERS_FOLDER"], mot_version, "data", "data")
        make_dir(mot_file_dir)
        tracker_file_name = str(scene_name) + ".txt"
        mot_file_name = os.path.join(mot_file_dir, tracker_file_name)
        write_dataframe_to_csv_to_file(mot_file_name, mot_pred_dataframe_per_scene)

        # Make sequence ini file for trackeval library
        if np.isnan(mot_pred_dataframe_per_scene["FrameId"].max()):
            last_frame_id = ground_truth_dataframe_per_scene["FrameId"].max()
        elif np.isnan(ground_truth_dataframe_per_scene["FrameId"].max()):
            last_frame_id = mot_pred_dataframe_per_scene["FrameId"].max()
        else:
            last_frame_id = max(mot_pred_dataframe_per_scene["FrameId"].max(), ground_truth_dataframe_per_scene["FrameId"].max())
        make_seq_ini_file(dir_name, scene=str(scene_name), seq_length=last_frame_id)          


    # Evaluate ground truth & prediction to get all exhaustive metrics
    evaluator = trackeval.eval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge3DLocation(dataset_config)]
    temp_metrics_list = [trackeval.metrics.HOTA]

    metrics_list = []
    for metric in temp_metrics_list:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))

    results = evaluator.evaluate(dataset_list, metrics_list)        

    if is_temp_dir:
        temp_dir.cleanup()
    return results

def evaluate(prediction_file: str, ground_truth_file: str, output_dir: str, num_cores: int, scene_2_camera_id_file: str) -> None:
    
    # Collect the result
    sequence_result = computes_mot_metrics(prediction_file, ground_truth_file, output_dir, num_cores, scene_2_camera_id_file)
    
    # Compute average
    final_result = dict()
    HOTA_scores = []
    DetA_scores = []
    AssA_scores = []
    LocA_scores = []
    for scene_name, result in sequence_result[0]["MotChallenge3DLocation"]["data"].items():
        
        if scene_name == "COMBINED_SEQ":
            continue
        result = result["pedestrian"]["HOTA"]
        HOTA_scores.append(np.mean(result["HOTA"]))
        DetA_scores.append(np.mean(result["DetA"]))
        AssA_scores.append(np.mean(result["AssA"]))
        LocA_scores.append(np.mean(result["LocA"]))

    final_result["FINAL"] = dict()
    final_result["FINAL"]["HOTA"] = np.mean(np.array(HOTA_scores))
    final_result["FINAL"]["DetA"] = np.mean(np.array(DetA_scores))
    final_result["FINAL"]["AssA"] = np.mean(np.array(AssA_scores))
    final_result["FINAL"]["LocA"] = np.mean(np.array(LocA_scores))
    return final_result

if __name__ == '__main__':
    start = time.time() 
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--prediction_file', required=True)
    parser.add_argument('--ground_truth_file', required=True)
    parser.add_argument('--output_dir')
    parser.add_argument('--num_cores', type=check_positive, default=1)
    parser.add_argument('--scene_2_camera_id_file', required=True)
    args = parser.parse_args()

    # Run evaluation
    final_result = evaluate(args.prediction_file, args.ground_truth_file, args.output_dir, args.num_cores, args.scene_2_camera_id_file)

    end = time.time()

    print(f"Total runtime: {end-start} seconds.")
    print(f"HOTA: {float(final_result['FINAL']['HOTA'] * 100):.4f}%")
    print(f"DetA: {float(final_result['FINAL']['DetA'] * 100):.4f}%")
    print(f"AssA: {float(final_result['FINAL']['AssA'] * 100):.4f}%")
    print(f"LocA: {float(final_result['FINAL']['LocA'] * 100):.4f}%")

