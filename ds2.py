import bpy
import os
import math
import shutil

# ---------- 配置区 ----------
OBJ_ROOT = "/home/oier/acgpj_change3d/objs"    # 替换为你的OBJ文件夹路径
OUTPUT_DIR = "/home/oier/acgpj_change3d/rendertry"  # 输出文件夹路径

# 相机参数（完全按照你的要求）
CAMERA_LOCATION = (150.0, -150.0, 250.0)
CAMERA_ROTATION = (
    math.radians(54.736),
    math.radians(-0.000003),
    math.radians(45.0)
)
CAMERA_CLIP_END = 500.0  # 裁剪距离500米

# 日光参数
SUN_ENERGY = 10.0

# 渲染参数
SAMPLES = 512
TARGET_ALPHA = 0.2  # 目标Alpha值

# ---------- 清理场景 ----------
def clean_scene():
    """只删除默认物体，保留必要元素"""
    to_remove = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name in ['Cube', 'Sphere', 'Plane']:
            to_remove.append(obj)
    
    for obj in to_remove:
        bpy.data.objects.remove(obj)
    
    print("✓ 场景清理完成")

# ---------- 关键修正：修改现有材质的Alpha值 ----------
def modify_materials_alpha(target_alpha=0.2):
    """
    修改场景中所有材质的Alpha值，但保留其他属性（颜色、纹理等）
    这会处理从.mtl文件导入的材质
    """
    materials_modified = 0
    
    for material in bpy.data.materials:
        # 跳过没有节点的材质
        print(material.name)
        # 如果名字中不包含Water
        if "Water" not in material.name:
            continue

        if not material.use_nodes:
            continue
            
        # 获取材质节点树
        nodes = material.node_tree.nodes
        
        # 查找原理化BSDF节点（从.mtl导入的材质通常有）
        bsdf_node = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
                break
        
        # 如果没有找到，尝试创建或查找其他类型节点
        if not bsdf_node:
            # 查找任何着色器节点
            for node in nodes:
                if 'BSDF' in node.type or 'Shader' in node.type:
                    bsdf_node = node
                    break
            
            # 如果还是没找到，创建一个新的原理化BSDF
            if not bsdf_node:
                bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # 修改Alpha值
        if hasattr(bsdf_node.inputs, 'get'):
            alpha_input = bsdf_node.inputs.get('Alpha')
            if alpha_input:
                alpha_input.default_value = target_alpha
                materials_modified += 1
                print(f"  已修改材质 '{material.name}' 的Alpha值为 {target_alpha}")
    
    return materials_modified

# ---------- 设置相机 ----------
def setup_camera():
    """按照精确参数设置相机"""
    # 删除现有相机
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)
    
    # 创建新相机
    cam_data = bpy.data.cameras.new("CameraData")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    
    # 设置变换
    cam_obj.location = CAMERA_LOCATION
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = CAMERA_ROTATION
    
    # 设置裁剪距离
    cam_obj.data.clip_end = CAMERA_CLIP_END
    
    # 设置为激活相机
    bpy.context.scene.camera = cam_obj
    
    print(f"✓ 相机设置完成")
    print(f"  位置: {CAMERA_LOCATION}")
    print(f"  裁剪结束: {CAMERA_CLIP_END}m")
    
    return cam_obj

# ---------- 设置日光 ----------
def setup_sunlight():
    """创建日光光源"""
    # 删除现有灯光
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj)
    
    # 创建日光
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun_obj = bpy.context.active_object
    sun_obj.name = "Daylight"
    sun_obj.data.energy = SUN_ENERGY
    
    # 设置日光角度使其照亮场景
    sun_obj.rotation_euler = (math.radians(50), math.radians(0), math.radians(45))
    
    print(f"✓ 日光设置完成")
    print(f"  强度: {SUN_ENERGY}")
    
    return sun_obj

# ---------- 设置渲染参数 ----------
def setup_render_settings():
    """配置渲染引擎"""
    scene = bpy.context.scene
    
    # 使用Cycles渲染引擎
    scene.render.engine = 'CYCLES'
    # 尝试启用GPU渲染（如果可用）
    try:
        enable_gpu_rendering(scene)
    except Exception as e:
        print(f"  ⚠ 无法自动启用GPU: {e}")
    
    # 采样设置
    scene.cycles.samples = SAMPLES
    
    # 透明背景
    # 使用白色背景：关闭透明背景，设置为RGB PNG，并设置世界背景为白色（适用于Cycles）
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'

    # 确保场景有一个World，并将背景设为白色
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg_node = None
    for node in nodes:
        if node.type == 'BACKGROUND':
            bg_node = node
            break
    if not bg_node:
        bg_node = nodes.new(type='ShaderNodeBackground')
    # 设置背景颜色为白色 (R, G, B, Alpha)
    try:
        bg_node.inputs['Color'].default_value = (0.1,0.1,0.1, 1.0)
    except Exception:
        pass
    
    # 分辨率
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    
    print(f"✓ 渲染设置完成")
    print(f"  采样: {SAMPLES}")
    print(f"  格式: PNG (RGBA)")


def enable_gpu_rendering(scene=None):
    """尝试启用 GPU 渲染：设置 compute device type、启用所有检测到的设备并切换到 GPU 计算。
    对不同 Blender 版本做了兼容处理（尽量不抛异常）。"""
    if scene is None:
        scene = bpy.context.scene

    prefs = bpy.context.preferences
    cycles_addon = prefs.addons.get('cycles')
    gpu_enabled = False

    if cycles_addon:
        prefs_cycles = cycles_addon.preferences
        # 优先尝试 OPTIX -> CUDA -> OPENCL
        for dev_type in ('OPTIX', 'CUDA', 'OPENCL'):
            try:
                prefs_cycles.compute_device_type = dev_type
                # 某些版本需要调用 get_devices() 来刷新设备列表
                try:
                    prefs_cycles.get_devices()
                except Exception:
                    try:
                        prefs_cycles.detect_devices()
                    except Exception:
                        pass
                # 如果设置成功，标记并退出循环
                gpu_enabled = True
                print(f"  尝试设置 GPU 计算类型: {dev_type}")
                break
            except Exception:
                continue

        # 尝试启用所有检测到的设备
        try:
            devices = prefs_cycles.devices
            for d in devices:
                try:
                    d.use = True
                except Exception:
                    pass
        except Exception:
            # 某些版本没有 devices 属性
            pass

    # 将渲染设备切换到 GPU（如果Cycles支持）
    try:
        scene.cycles.device = 'GPU'
        gpu_enabled = True
    except Exception:
        try:
            # 旧API：
            bpy.context.scene.render.device = 'GPU'
            gpu_enabled = True
        except Exception:
            pass

    # 根据是否启用GPU调整tile大小和持久数据
    try:
        if gpu_enabled:
            # GPU 通常使用较大的 tile
            try:
                scene.render.tile_x = 256
                scene.render.tile_y = 256
            except Exception:
                pass
            try:
                scene.cycles.use_persistent_data = True
            except Exception:
                pass
            print("✓ 已尝试启用 GPU 渲染（若设备可用），并调整 tile 为 256x256")
        else:
            # 回退到 CPU 推荐较小 tile
            try:
                scene.render.tile_x = 32
                scene.render.tile_y = 32
            except Exception:
                pass
            print("i GPU 未启用，继续使用 CPU 渲染")
    except Exception:
        pass


# ---------- 导入OBJ并处理材质 ----------
def import_and_render_obj(obj_path, output_path):
    """
    导入单个OBJ文件（自动加载.mtl），修改Alpha值，然后渲染
    """
    # 清理之前导入的网格
    mesh_objects = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_objects.append(obj)
    
    for obj in mesh_objects:
        bpy.data.objects.remove(obj)
    
    print(f"  正在导入: {os.path.basename(obj_path)}")
    
    # 关键：使用Blender的OBJ导入器（会自动加载同目录下的.mtl文件）
    try:
        # 这是Blender 3.0+的OBJ导入操作
        bpy.ops.wm.obj_import(
            filepath=obj_path,
            forward_axis='Z',  # 可能需要根据你的模型调整
            up_axis='Y'
        )# XZ YX XY ZX ZY√
        
        # 检查导入的网格
        imported_meshes = []
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                imported_meshes.append(obj)
                print(f"    导入网格: {obj.name}")
                
                # 打印材质信息
                if obj.data.materials:
                    for mat in obj.data.materials:
                        print(f"      使用材质: {mat.name}")
        
        if not imported_meshes:
            print(f"  ⚠ 未找到网格数据")
            return False
        
        # 修改所有材质的Alpha值（关键步骤）
        print(f"  修改材质Alpha值...")
        materials_modified = modify_materials_alpha(TARGET_ALPHA)
        print(f"    修改了 {materials_modified} 个材质的Alpha值")
        
        # 设置输出路径并渲染
        bpy.context.scene.render.filepath = output_path
        print(f"  开始渲染...")
        bpy.ops.render.render(write_still=True)
        
        return True
        
    except Exception as e:
        print(f"  ✗ 导入或渲染失败: {e}")
        return False

# ---------- 主函数 ----------
def main():
    print("=" * 60)
    print("Blender批量渲染 - 支持MTL材质")
    print("=" * 60)
    
    # 检查路径
    if not os.path.exists(OBJ_ROOT):
        print(f"✗ 错误: 文件夹不存在 - {OBJ_ROOT}")
        return
    
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ 创建输出目录: {OUTPUT_DIR}")
    
    # 初始设置（只在开始时执行一次）
    print("\n1. 初始设置")
    print("-" * 40)
    clean_scene()
    setup_camera()
    setup_sunlight()
    setup_render_settings()
    
    # 查找所有OBJ文件
    obj_files = []
    for file in os.listdir(OBJ_ROOT):
        if file.lower().endswith('.obj'):
            obj_files.append(file)
    
    if not obj_files:
        print(f"\n✗ 没有找到OBJ文件")
        return
    
    print(f"\n2. 找到 {len(obj_files)} 个OBJ文件")
    print("-" * 40)
    
    # 处理每个OBJ文件
    success_count = 0
    
    for i, filename in enumerate(obj_files, 1):
        print(f"\n[{i}/{len(obj_files)}] 处理: {filename}")
        
        obj_path = os.path.join(OBJ_ROOT, filename)
        
        # 检查对应的.mtl文件是否存在
        mtl_name = os.path.splitext(filename)[0] + '.mtl'
        mtl_path = os.path.join(OBJ_ROOT, mtl_name)
        
        if os.path.exists(mtl_path):
            print(f"  找到MTL材质文件: {mtl_name}")
        else:
            print(f"  ⚠ 警告: 未找到对应的MTL文件")
        
        # 输出路径
        output_name = f"{os.path.splitext(filename)[0]}_render.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # 导入和渲染
        if import_and_render_obj(obj_path, output_path):
            success_count += 1
            print(f"  ✓ 完成: {output_name}")
        else:
            print(f"  ✗ 失败: {filename}")
    
    # 总结
    print("\n" + "=" * 60)
    print("批量渲染完成!")
    print("-" * 60)
    print(f"总计: {len(obj_files)} 个文件")
    print(f"成功: {success_count} 个")
    print(f"失败: {len(obj_files) - success_count} 个")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

# ---------- 脚本入口 ----------
if __name__ == "__main__":
    main()