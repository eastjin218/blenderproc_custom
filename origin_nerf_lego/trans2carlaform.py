import json, os, glob
import numpy as np
from scipy.spatial.transform import Rotation as R

json_path = '/home/BlenderProc/custom_data/ori_val.json'
with open(json_path, 'r') as f:
    json_data = json.load(f)

frames = json_data['frames']
with open('./camera_val_trans.json', 'w') as f:
    result = []
    for i in frames:
        sub_result={}
        trans_matrix = np.array(i['transform_matrix'])
        rot = trans_matrix[:3,:3]
        xyz_dis = trans_matrix[:3,3]
        rot_mat = R.from_matrix(rot)
        x_deg, y_deg, z_deg = rot_mat.as_euler('xyz', degrees=True)
        sub_result['angle'] = z_deg
        sub_result['pitch'] = x_deg
        sub_result['distance'] = 15
        result.append(sub_result)
    json.dump(result, f, indent=2)