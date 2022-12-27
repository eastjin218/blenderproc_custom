import blenderproc as bproc
import bpy
import bmesh
from pathlib import Path
from blenderproc.python.types.MaterialUtility import Material
import numpy as np
bproc.init()

objs = bproc.loader.load_obj('./m1a2.obj')
for obj in objs:
    obj.set_location([0,0,-1])
    obj.set_rotation_euler([np.pi/2, 0, 0])
    obj.add_uv_mapping('cylinder', True)
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
    obj.object_mode() # mesh edit mode convert to object mode
    print(obj.get_bound_box()) # 객체 3D bounding box 좌표 (x,y,z) 8개
    # print(obj.get_bound_box_volume()) # 갹체 볼륨
    obj.scale_uv_coordinates(1/0.5) # texture의 unit scale 배율 조절 (1/2) => texture scale 2m /기본은 1m

light = bproc.types.Light()
light.set_type("SUN")
light.set_distance(100)
light.set_energy(10)

bproc.camera.set_resolution(1024, 720)

bproc.camera.add_camera_pose(
  bproc.math.build_transformation_mat(
       [0, -20, 0], [np.pi/2,  -0, 0]))
    # [-5.24497166, 13.32843661, 4.83308047], [1.24536524e+00,  -2.29932344e-08, -2.76668719e+00]))

materials = bproc.material.collect_all()

# image_path = '/home/BlenderProc/custom_data/img_16.png'
image_path ='/home/BlenderProc/custom_data/dta_best_attack_pattern.png'

texture = bpy.data.images.load(str(image_path), check_existing=True)
for mat in materials:
    mat.set_principled_shader_value("Base Color", texture)
# materials[2].set_principled_shader_value("Base Color", texture)

bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)

data = bproc.renderer.render()

bproc.writer.write_hdf5('./triplanar', data)