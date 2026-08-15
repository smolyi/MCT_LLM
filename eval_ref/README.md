# Evaluation Code for - MTMC Tracking 2024 Dataset

Evaluation code for the Multi-Target Multi-Camera (MTMC) 2024 dataset. 

The evaluation utilizes Higher Order Tracking Accuracy (HOTA) score as an evaluation metric for multi-object tracking that addresses the limitations of previous metrics like MOTA and IDF1. It integrates three key aspects of MOT: accurate detection, association, and localization into a unified metric. This comprehensive approach balances the importance of detecting each object (detection), correctly identifying objects across different frames (association), and accurately localizing objects in each frame (localization). Furthermore, HOTA can be decomposed into simpler components, allowing for detailed analysis of different aspects of tracking behavior. 

HOTA scores calculated using 3D distance measurements in a multi-camera setting.

# Environment setup:
```
- [Optional] conda create -n mtmc_eval_2024 python=3.10
- [Optional] conda activate mtmc_eval_2024

- pip3 install pandas
- pip3 install matplotlib
- pip3 install scipy
```

# Usage: 

 - Set the --prediction_file argument to a valid prediction file.
 - Set the --ground_truth_file argument to a valid test file.
 - Set the --num_cores argument based on your setup.
 - Set the --scene_2_camera_id_file argument 


Example below:
```
python3 main.py --prediction_file ./sample_file/pred.txt --ground_truth_file ./sample_file/ground_truth_test_full.txt --num_cores 16 --scene_2_camera_id_file ./sample_file/scene_name_2_cam_id_full.json

Sample Result:
Total runtime: 187.0887589454651 seconds.
HOTA: 49.2825%
DetA: 49.1998%
AssA: 49.3655%
LocA: 77.0546%
```

## Acknowledgements

This project utilizes a portion of code from [TrackEval](https://github.com/JonathonLuiten/TrackEval), an open-source project by Jonathon Luiten for evaluating multi-camera tracking results. TrackEval is licensed under the MIT License, which you can find in full [here](https://github.com/JonathonLuiten/TrackEval/blob/master/LICENSE).