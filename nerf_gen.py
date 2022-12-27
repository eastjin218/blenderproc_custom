import blenderproc as bproc
import bpy
import bmesh
import argparse
import json
import os, glob, shutil
import numpy as np

from pathlib import Path
from blenderproc.python.types.MaterialUtility import Material

def define_config():
    p = argparse.ArgumentParser()

    p.add_argument(
        '-i',
        '--object_path',
        help='object file path',
        default='./m1a2.obj'
    )
    p.add_argument(
        '-c',
        '--camera_json_path',
        help='camera transform file path',
        default='./camera_train_trans.json'
    )
    p.add_argument(
        '-o',
        '--output_path',
        help ='hdf5 file save path',
        default='./output'
    )
    p.add_argument(
        '-t',
        '--texture_path',
        help='rendering texture path',
        default='/home/BlenderProc/custom_data/dta_best_attack_pattern.png',
    )
    p.add_argument(
        '-ss',
        '--shard_size',
        help='shard size of rendering data',
        type=int,
        default=100,
    )
    p.add_argument(
        '-ts',
        '--texture_scale',
        help ='texture scale of object',
        type = float,
        default = 1.28,
    )
    p.add_argument(
        '--use_texture',
        help='texture using control',
        action='store_true'
    )
    config = p.parse_args()
    return config 

def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def object_setup(
    objs, config, location=[0,0,-1], rotation=[np.pi/2, 0,np.pi/2+np.pi]):
    for obj in objs:
        obj.set_location(location)
        obj.set_rotation_euler(rotation)
        obj.edit_mode()
        obj_bm = obj.mesh_as_bmesh()
        uv_layer = obj_bm.loops.layers.uv.verify()
        # obj_bm.faces.layers.tex.verify()  # currently blender needs both layers.
        for f in obj_bm.faces:
            norm = f.normal
            ax, ay, az = abs(norm.x), abs(norm.y), abs(norm.z)
            axis = -1
            if ax > ay and ax > az:
                axis = 0
            if ay > ax and ay > az:
                axis = 1
            if az > ax and az > ay:
                axis = 2
            for l in f.loops:
                luv = l[uv_layer]
                if axis == 0: # x plane     
                    luv.uv.x = l.vert.co.y
                    luv.uv.y = l.vert.co.z
                if axis == 1: # u plane
                    luv.uv.x = l.vert.co.x
                    luv.uv.y = l.vert.co.z
                if axis == 2: # z plane
                    luv.uv.x = l.vert.co.x
                    luv.uv.y = l.vert.co.y
        obj.update_from_bmesh(obj_bm)
        obj.object_mode()
        obj.scale_uv_coordinate(1/config.texture_scale)


def light_setup(
    light, location=[5,0,5], energy=1000
):
    light.set_type("AREA")
    # light.set_distance(100000)
    light.set_location(location)
    light.set_energy(energy)

def get_camera_param(angle, pitch, distance):
    z = distance * np.sin(np.radians(90-pitch))
    xy_dis = distance * np.cos(np.radians(90-pitch))
    x = xy_dis * np.sin(np.radians(angle))
    y = xy_dis * np.cos(np.radians(angle))
    trans = [x, -1*y, z]
    rot = [np.radians(pitch), np.radians(0), np.radians(angle)]
    return trans, rot

def camera_setup(
    json_config=None, ty='train',start_num=0, end_num=100, batch=0, 
    shard_size=100, resolution=(1024,720)
):
    res_x = resolution[0]
    res_y = resolution[1]
    bproc.camera.set_resolution(res_x, res_y)
    fov_x, fov_y = bproc.camera.get_fov()
    transform_json = {}
    with open(f'transform_{ty}_{batch}.json', 'w') as f:
        transform_json['camera_angle_x']=fov_x
        transform_json['camera_angle_y']=fov_y
        transform_json['w']=res_x
        transform_json['h']=res_y
        transform_json['fl_x']=.5 * res_x / np.tan(.5 * fov_x)
        transform_json['fl_y']=.5 * res_y / np.tan(.5 * fov_y)
        transform_json['frames']=[]

        for idx, j_config in enumerate(json_config[start_num:end_num]):
            idx = (batch*shard_size) +idx
            sub_trans_json={}
            angle = j_config['angle']
            pitch = j_config['pitch']
            distance = j_config['distance']
            trans, rot = get_camera_param(angle, pitch, distance)
            matrix = bproc.math.build_transformation_mat(trans, rot)
            bproc.camera.add_camera_pose(matrix)

            matrix[:3,3] = matrix[:3,3]*4/distance
            sub_trans_json['file_path']=f'{ty}/{idx}_colors'
            sub_trans_json['rotation']=0.01256
            sub_trans_json['transform_matrix']=matrix.tolist()
            transform_json['frames'].append(sub_trans_json)
        json.dump(transform_json, f, indent=2)
        
def rendering_texture(materials, texture_path):
    texture = bpy.data.images.load(texture_path, check_existing=True)
    for mat in materials:
        mat.set_principled_shader_value("Base Color", texture)

def sort_file(output_path, shard_size=100):
    files = glob.glob(output_path+'_*/*')
    files.sort()
    os.makedirs(output_path, exist_ok=True)
    remove = []
    for fi in files:
        dn = os.path.dirname(fi)
        if dn not in remove:
            remove.append(dn)
        pre_idx = dn.split('_')[-1]
        batch = shard_size * int(pre_idx)
        fn = str(int(os.path.basename(fi).split('.')[0]) +batch)+'.hdf5'
        shutil.copyfile(fi, os.path.join(output_path, fn))
    for i in remove:
        shutil.rmtree(i)

def merg_json(ty='train'):
    json_files = glob.glob(f'./transform_{ty}_*')
    with open(json_files[0], 'r') as f:
        json_data = json.load(f)
    for i in json_files[1:]:
        with open(i, 'r') as k:
            data = json.load(k)
        json_data['frames'].extend(data['frames'])
    with open(f'transforms_{ty}.json','w') as i:
        json.dump(json_data, i, indent=2)
    for j in json_files:
        os.remove(j)


def main(config):
    camera_json = load_json(config.camera_json_path)
    bproc.init()
    ty = config.camera_json_path.split('_')[-1].split('.')[0]
    if os.path.isdir(config.output_path):
        shutil.rmtree(config.output_path)
    for batch in range(len(camera_json)//config.shard_size):
        start_idx = batch*config.shard_size
        end_idx = (batch+1)*config.shard_size
        
        objs = bproc.loader.load_obj(config.object_path)
        object_setup(objs, config)
        light = bproc.types.Light()
        light_setup(light)
        camera_setup(camera_json, ty,
            start_num=start_idx, end_num=end_idx, batch=int(batch), shard_size=config.shard_size)
        if config.use_texture:
            print('using Texture!!')
            materials = bproc.material.collect_all()
            rendering_texture(materials, config.texture_path)
        data = bproc.renderer.render()
        bproc.writer.write_hdf5(config.output_path+'_'+str(batch), data)
        bproc.clean_up()
    sort_file(config.output_path, config.shard_size)
    merg_json(ty)

if __name__=='__main__':
    config = define_config()
    main(config)