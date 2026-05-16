import bpy
import sys
import numpy as np
import mathutils
import shutil
import os
import argparse
import random

random.seed(42)
np.random.seed(42)


def parse_args():
    argv = sys.argv
    if "--args" in argv:
        argv = argv[argv.index("--args") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="cameras.npy")
    parser.add_argument("--mesh", type=str, default="head.ply")
    parser.add_argument("--hair", type=str, default=None)
    parser.add_argument("--save_output", type=str, default="save_out")
    parser.add_argument("--save_name", type=str, default="render_default")
    args = parser.parse_args(argv)
    return args

def enable_gpus():
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    for device in list(cycles_preferences.devices)[:2]:
        device.use = True
    cycles_preferences.compute_device_type = 'OPTIX'
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_persistent_data = True

def rf_rq(P):
    P = P.T
    q, r = np.linalg.qr(P[::-1, ::-1], 'complete')
    return r.T[::-1, ::-1], q.T[::-1, ::-1]

def KRT_from_P(P):
    H = P[:, :3]
    K, R = rf_rq(H)
    K /= K[-1, -1]
    sg = np.diag(np.sign(np.diag(K)))
    K = K @ sg
    R = sg @ R
    if np.linalg.det(R) < 0: R = -R
    C = np.linalg.lstsq(-H, P[:, -1], rcond=None)[0]
    T = -R @ C
    return K, R, T

def get_blender_camera_from_3x4_P(P, scale, name):
    K, R, T = KRT_from_P(np.array(P))
    scene = bpy.context.scene

    sensor_width = K[1,1]*K[0,2]/(K[0,0]*K[1,2])
    res_x, res_y = int(K[0,2] * 2), int(K[1,2] * 2)
    s_u = res_x / sensor_width
    f_mm = K[0,0] / s_u

    scene.render.resolution_x = res_x // scale
    scene.render.resolution_y = res_y // scale
    scene.render.resolution_percentage = scale * 100

    R_bcam2cv = mathutils.Matrix(((1,0,0),(0,-1,0),(0,0,-1)))
    R_cv2world = mathutils.Matrix(R.T.tolist())
    rotation = R_cv2world @ R_bcam2cv
    location = R_cv2world @ (-mathutils.Vector(T.tolist()))

    bpy.ops.object.add(type='CAMERA', location=location)
    ob = bpy.context.object
    ob.name = name
    cam = ob.data
    cam.name = name
    cam.type = 'PERSP'
    cam.lens = f_mm
    cam.sensor_width = sensor_width
    ob.matrix_world = mathutils.Matrix.Translation(location) @ rotation.to_4x4()
    return ob

def set_material(obj, material):
    if len(obj.material_slots) < 1:
        obj.data.materials.append(material)
    else:
        obj.material_slots[obj.active_material_index].material = material

def create_hair_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    links.clear()
    output = nodes.new(type='ShaderNodeOutputMaterial')
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')

    links.new(shader.outputs[0], output.inputs[0])
    shader.inputs[0].default_value = color
    shader.inputs[7].default_value = 0.0
    return mat

def create_hair(name, strands, color):
    curve = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 1

    for strand in strands:
        spline = curve.splines.new('POLY')
        spline.points.add(len(strand) - 1)
        for i, (x, y, z) in enumerate(strand):
            spline.points[i].co = (x, y, z, 1)

    obj = bpy.data.objects.new(name, curve)
    bpy.data.scenes[0].collection.objects.link(obj)
    obj.rotation_euler[0] = -np.pi / 2
    mat = create_hair_material(f'{name}_mat', color)
    set_material(obj, mat)

    # thickness of the curves
    obj.data.bevel_depth =  0.0001 #0.001
    return obj



def main():
    args = parse_args()
    enable_gpus()

    cameras = np.load(args.camera)
    camera_obs = [get_blender_camera_from_3x4_P(c, 1, str(i)) for i, c in enumerate(cameras)]

    bpy.ops.import_mesh.ply(filepath=args.mesh)
    head_obj = bpy.context.selected_objects[0]
    head_name = head_obj.name
    set_material(head_obj, bpy.data.materials['Main'])

    if args.hair and os.path.exists(args.hair):
        hair = np.load(args.hair)
        n_strands = len(hair)
        blocks = 4
        colors = [(0.125, 0.5, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0), (0.125, 0.0, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0)]
        hair = hair[np.random.choice(n_strands, n_strands, replace=False)]
        for i in range(blocks):
            chunk = hair[i * (n_strands // blocks):(i + 1) * (n_strands // blocks)]
            create_hair(f'Hair_{i}', chunk, colors[i])

    if 'placeholder' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['placeholder'])

    for scene in bpy.data.scenes:
        scene.render.resolution_x = 1920 #1024
        scene.render.resolution_y = 1920 # 1024

    # Create output directory per head
    #out_dir = os.path.join("render_results", args.save_name, args.hair.split('/')[-1].split('.')[0])
    
    dir_path=os.path.join(args.save_output, "render_results")
    os.makedirs(dir_path, exist_ok=True)

    out_dir = os.path.join(dir_path, args.hair.split('/')[-1].split('.')[0])
    os.makedirs(out_dir, exist_ok=True)

    for i, cam in enumerate(camera_obs):
        bpy.context.scene.camera = cam
        bpy.context.scene.cycles.samples = 1024
        bpy.context.view_layer.update()
        bpy.context.scene.render.filepath = os.path.join(out_dir, f"{i:06d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
