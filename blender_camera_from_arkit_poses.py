import bpy
import math
import mathutils
import time
import numpy as np
from mathutils import Matrix
from random import randint
import os
import json


def load_frames(json_base_path):
    
    json_files = list(sorted(filter(lambda s : 'frame_' in s and '.json' in s, os.listdir(json_base_path))))
    json_files = list(map(lambda s: os.path.join(json_base_path, s) , json_files))
    
    frames = []
    
    for json_file in json_files:
        info = json.load(open(json_file))
        pose = np.array( info["cameraPoseARFrame"] ).reshape((4,4))
        timestamp = info.get("time", -1)
        frame = dict(pose=pose.copy(), time=timestamp)
        frames.append(frame)
    
    first_time = frames[0]["time"]
    
    if first_time >= 0:
        for f in frames:
            f["time"] -= first_time     
    
    return frames 


def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()



def eraseAllKeyframes(scene, passedOB = None):

    if passedOB != None:

        ad = passedOB.animation_data

        if ad != None:
            print ('ad=',ad)
            passedOB.animation_data_clear()

            #scene.update()
            

def insert_keyframes(obj, frames, scene_fps=30.0, frame_offset=0):
    
    # insert animation keyframes for the given 'obj' for each frame pose
    
    eraseAllKeyframes(scene, obj)
    
    for i in range(len(frames)):
        
        frame = frames[i]
        
        pose = frame['pose']
            
        y_rows = pose[1, :].copy()
        z_rows = pose[2, :].copy()
        
        pose[1, :] = z_rows * -1.0
        pose[2, :] = y_rows 
                
        frame_idx = round(frame["time"] * float(scene_fps))
        frame_idx += frame_offset
        
        bpy.context.scene.frame_set(frame_idx)
        
        obj.matrix_world = Matrix(pose)
        
        obj.keyframe_insert('rotation_euler', index=-1)
        obj.keyframe_insert('location', index=-1)
        

    return frame_idx


#############################################

# Set scan path 
scan_path = "/Users/cc/Downloads/2022_01_09_12_54_15/optimized_poses"

frames = load_frames(scan_path)

print("Loaded frames: " , len(frames))
            
scene = bpy.data.scenes["Scene"]
cam = bpy.data.objects['Camera']

scene.frame_start = 0

# blender camera looks down -Z , +Y is up , +X right 

last_frame_idx = insert_keyframes(cam, frames, frame_offset=0)

bpy.context.scene.frame_set(0)

scene.frame_end = last_frame_idx

