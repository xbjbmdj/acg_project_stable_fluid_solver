import bpy
import os
import mathutils
import math

# ---------- 配置区：所有参数可在此调整 ----------
OBJ_ROOT = "/home/oier/acgpj_change3d/objs"      # 替换为你的OBJ文件夹路径
OUTPUT_DIR = "/home/oier/acgpj_change3d/outputs"            # 替换为输出文件夹路径

# 相机参数（完全按照你的图片设置）
CAMERA_LOCATION = (200.0, -200.0, 200.0)  # 位置 (X, Y, Z)
CAMERA_ROTATION = (                       # 旋转 (X, Y, Z) 单位：度
    math.radians(54.736),                 # X轴：54.736°
    math.radians(-0.000003),              # Y轴：-0.000003°
    math.radians(45.0)                    # Z轴：45°
)
CAMERA_CLIP_START = 0.1                   # 裁剪起始（默认值）
CAMERA_CLIP_END = 500.0                   # 裁剪结束：500米

# 日光参数
SUN_ENERGY = 10.0                         # 日光强度：10.0
SUN_ROTATION = (                          # 日光旋转（指向下方）
    math.radians(45.0),                   # 适当角度照亮场景
    math.radians(0.0),
    math.radians(0.0)
)

# 渲染参数
SAMPLES = 512                             # 采样次数：512
ALPHA_VALUE = 0.2                         # 材质Alpha值：0.2
RESOLUTION = (1920, 1080)                 # 输出分辨率

# ---------- 清理场景 ----------
def clean_scene():
    """只删除可能干扰的默认物体，保留必要元素"""
    objects_to_remove = []
    
    for obj in bpy.context.scene.objects:
        # 删除默认的起始物体，但保留可能的手动添加物体
        if (obj.type == 'MESH' and 
            obj.name in ['Cube', 'Sphere', 'Plane', 'Cylinder', 'Cone', 'Torus']):
            objects_to_remove.append(obj)
    
    # 批量删除
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    print("✓ 场景清理完成")
    return len(objects_to_remove)

# ---------- 创建透明材质 ----------
def create_transparent_material(alpha=0.2):
    """创建指定Alpha值的透明材质"""
    mat = bpy.data.materials.new(name=f"Transparent_Alpha{alpha}")
    mat.use_nodes = True
    
    if mat.node_tree:
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # 创建原理化BSDF节点
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs['Alpha'].default_value = alpha
        
        # 创建材质输出节点
        output = nodes.new(type='ShaderNodeOutputMaterial')
        
        # 连接节点
        links = mat.node_tree.links
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    print(f"✓ 创建材质: Alpha = {alpha}")
    return mat

# ---------- 精确设置相机 ----------
def setup_camera():
    """按照提供的精确参数设置相机"""
    scene = bpy.context.scene
    
    # 检查是否已有相机
    existing_cameras = [obj for obj in scene.objects if obj.type == 'CAMERA']
    
    if existing_cameras:
        cam_obj = existing_cameras[0]
        cam_obj.name = "RenderCamera"
        print(f"✓ 使用现有相机: {cam_obj.name}")
    else:
        # 创建新相机
        cam_data = bpy.data.cameras.new("CameraData")
        cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        print("✓ 创建新相机")
    
    # 设置变换参数（完全匹配图片）
    cam_obj.location = CAMERA_LOCATION
    cam_obj.rotation_mode = 'XYZ'  # 设置为XYZ欧拉模式
    cam_obj.rotation_euler = CAMERA_ROTATION
    
    # 设置裁剪距离
    cam_obj.data.clip_start = CAMERA_CLIP_START
    cam_obj.data.clip_end = CAMERA_CLIP_END
    
    # 设置为场景激活相机
    scene.camera = cam_obj
    
    # 验证设置
    loc = cam_obj.location
    rot = [math.degrees(r) for r in cam_obj.rotation_euler]
    
    print(f"  位置: X={loc.x:.3f}m, Y={loc.y:.3f}m, Z={loc.z:.3f}m")
    print(f"  旋转: X={rot[0]:.6f}°, Y={rot[1]:.6f}°, Z={rot[2]:.6f}°")
    print(f"  模式: {cam_obj.rotation_mode} 欧拉")
    print(f"  裁剪: {CAMERA_CLIP_START} - {CAMERA_CLIP_END}m")
    
    return cam_obj

# ---------- 设置日光 ----------
def setup_sunlight():
    """创建并设置日光光源"""
    # 检查是否已有日光
    existing_suns = [obj for obj in bpy.context.scene.objects 
                    if obj.type == 'LIGHT' and obj.data.type == 'SUN']
    
    if existing_suns:
        sun_obj = existing_suns[0]
        sun_obj.name = "Daylight"
        print(f"✓ 使用现有日光: {sun_obj.name}")
    else:
        # 创建新日光
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun_obj = bpy.context.active_object
        sun_obj.name = "Daylight"
        print("✓ 创建新日光")
    
    # 设置日光参数
    sun_obj.data.energy = SUN_ENERGY
    sun_obj.rotation_euler = SUN_ROTATION
    
    print(f"  强度: {SUN_ENERGY}")
    print(f"  角度: {[math.degrees(r) for r in SUN_ROTATION]}")
    
    return sun_obj

# ---------- 设置渲染参数 ----------
def setup_render_settings():
    """配置渲染引擎和输出设置"""
    scene = bpy.context.scene
    render = scene.render
    
    # 渲染引擎
    render.engine = 'CYCLES'
    
    # 采样设置
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = False  # 固定采样
    
    # 透明背景
    render.film_transparent = True
    render.image_settings.color_mode = 'RGBA'
    render.image_settings.file_format = 'PNG'
    
    # 分辨率
    render.resolution_x = RESOLUTION[0]
    render.resolution_y = RESOLUTION[1]
    render.resolution_percentage = 100
    
    print(f"✓ 渲染设置完成")
    print(f"  引擎: {render.engine}")
    print(f"  采样: {SAMPLES}")
    print(f"  透明: {render.film_transparent}")
    print(f"  格式: PNG (RGBA)")
    print(f"  分辨率: {RESOLUTION[0]}x{RESOLUTION[1]}")
    
    return True

# ---------- 导入并处理OBJ ----------
def import_and_process_obj(obj_path, material):
    """导入单个OBJ文件并应用材质"""
    # 清除之前导入的网格
    mesh_objects = [obj for obj in bpy.context.scene.objects 
                   if obj.type == 'MESH' and not obj.name.startswith(('Daylight', 'RenderCamera'))]
    
    for obj in mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # 导入OBJ
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
        
        # 获取导入的网格
        imported_meshes = [obj for obj in bpy.context.selected_objects 
                          if obj.type == 'MESH']
        
        if not imported_meshes:
            print(f"  ⚠ 警告: 未找到网格数据")
            return []
        
        # 应用材质
        for obj in imported_meshes:
            obj.data.materials.clear()
            obj.data.materials.append(material)
        
        print(f"  ✓ 导入成功: {len(imported_meshes)} 个网格")
        return imported_meshes
        
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return []

# ---------- 主函数 ----------
def main():
    print("=" * 60)
    print("Blender 批量渲染自动化脚本")
    print("=" * 60)
    
    # 检查路径
    if not os.path.exists(OBJ_ROOT):
        print(f"✗ 错误: OBJ文件夹不存在 - {OBJ_ROOT}")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ 创建输出目录: {OUTPUT_DIR}")
    
    # 初始设置
    print("\n1. 初始化设置")
    print("-" * 40)
    
    cleaned = clean_scene()
    if cleaned > 0:
        print(f"  已删除 {cleaned} 个默认物体")
    
    material = create_transparent_material(ALPHA_VALUE)
    setup_camera()
    setup_sunlight()
    setup_render_settings()
    
    # 获取OBJ文件列表
    obj_files = sorted([f for f in os.listdir(OBJ_ROOT) 
                       if f.lower().endswith('.obj')])
    
    if not obj_files:
        print(f"\n✗ 错误: 在 {OBJ_ROOT} 中没有找到OBJ文件")
        return
    
    print(f"\n2. 找到 {len(obj_files)} 个OBJ文件")
    print("-" * 40)
    
    # 批量处理
    success_count = 0
    
    for i, filename in enumerate(obj_files, 1):
        print(f"\n[{i}/{len(obj_files)}] 处理: {filename}")
        
        obj_path = os.path.join(OBJ_ROOT, filename)
        
        # 导入并处理OBJ
        meshes = import_and_process_obj(obj_path, material)
        
        if not meshes:
            print(f"  ⚠ 跳过此文件")
            continue
        
        # 设置输出路径
        output_name = f"{os.path.splitext(filename)[0]}_render.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        bpy.context.scene.render.filepath = output_path
        
        # 渲染
        print(f"  ↪ 渲染中...")
        try:
            bpy.ops.render.render(write_still=True)
            success_count += 1
            print(f"  ✓ 渲染完成: {output_name}")
        except Exception as e:
            print(f"  ✗ 渲染失败: {e}")
        
        # 清理当前模型
        for obj in meshes:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # 总结报告
    print("\n" + "=" * 60)
    print("批量渲染完成!")
    print("-" * 60)
    print(f"处理总数: {len(obj_files)} 个文件")
    print(f"成功渲染: {success_count} 个文件")
    print(f"失败/跳过: {len(obj_files) - success_count} 个文件")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

# ---------- 脚本入口 ----------
if __name__ == "__main__":
    # 在命令行中运行（无界面模式，推荐）
    # blender --background --python 此脚本.py
    
    main()