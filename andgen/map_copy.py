import os
import shutil
import multiprocessing
from tqdm import tqdm

# 源目录路径
src_base = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss/anp_shard-100_samples-5000/train"
# 目标根目录
dst_root = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss"

# 全景图文件名前缀
panorama_prefix = "map-"

def process_scene(scene):
    scene_path = os.path.join(src_base, scene)
    
    # 查找场景的全景图文件
    panorama_files = [f for f in os.listdir(scene_path) if f.startswith(panorama_prefix) and f.endswith(".png")]
    
    if not panorama_files:
        print(f"Warning: No panorama file found in scene {scene}")
        return
    
    panorama_file = panorama_files[0]  # 假设只有一个全景图文件
    
    # 处理50个分片
    for shard_idx in range(50):
        # 计算目标目录的n值（从2开始）
        n = shard_idx + 2
        dst_dir = os.path.join(
            dst_root,
            f"anp_shard-{n}_samples-100",
            "train",
            scene
        )
        
        # 创建目标目录（如果不存在）
        os.makedirs(dst_dir, exist_ok=True)
        
        # 源文件路径
        src_path = os.path.join(scene_path, panorama_file)
        
        # 目标文件路径
        dst_path = os.path.join(dst_dir, panorama_file)
        
        # 复制文件
        shutil.copy2(src_path, dst_path)
    
    return scene

def main():
    # 获取所有场景文件夹
    scenes = [d for d in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, d))]
    
    print(f"Found {len(scenes)} scenes to process")
    
    # 使用进程池处理（最多5个进程）
    results = []
    with multiprocessing.Pool(processes=5) as pool:
        # 使用tqdm显示进度条
        with tqdm(total=len(scenes), desc="Processing scenes") as pbar:
            for result in pool.imap_unordered(process_scene, scenes):
                if result:
                    results.append(result)
                pbar.update(1)
    
    print("\nProcessing completed!")
    print(f"Successfully processed {len(results)} scenes")
    print(f"Copied panorama images to {50 * len(results)} shard directories")

if __name__ == "__main__":
    main()