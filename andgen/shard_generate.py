import os
import shutil
import multiprocessing
from tqdm import tqdm

# 源目录路径
src_base = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss/anp_shard-100_samples-5000/train"
# 目标根目录
dst_root = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss"

# 定义样本文件的后缀
file_suffixes = ["-ir_receiver.wav", "-rgb.png", "-depth.png"]

def process_scene(scene):
    scene_path = os.path.join(src_base, scene)
    
    # 获取所有样本ID（数字前缀）
    sample_ids = set()
    for fname in os.listdir(scene_path):
        if any(fname.endswith(suffix) for suffix in file_suffixes):
            sample_id = fname.split("-")[0]
            try:
                sample_ids.add(int(sample_id))
            except ValueError:
                continue
    
    # 转换为列表并排序
    sample_ids = sorted(list(sample_ids))
    if not sample_ids:
        print(f"Warning: No samples found in scene {scene}")
        return
    
    # 计算每个分片的样本范围
    num_shards = 50
    samples_per_shard = len(sample_ids) // num_shards
    
    # 处理每个分片
    for shard_idx in range(num_shards):
        # 计算当前分片的样本范围
        start_idx = shard_idx * samples_per_shard
        end_idx = (shard_idx + 1) * samples_per_shard
        shard_samples = sample_ids[start_idx:end_idx]
        
        # 计算目标目录
        n = shard_idx + 2
        dst_dir = os.path.join(dst_root, f"anp_shard-{n}_samples-100", "train", scene)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 复制该分片的所有样本
        for sample_id in shard_samples:
            for suffix in file_suffixes:
                src_file = os.path.join(scene_path, f"{sample_id}{suffix}")
                dst_file = os.path.join(dst_dir, f"{sample_id}{suffix}")
                
                # 跳过已存在的文件
                if os.path.exists(dst_file):
                    continue
                
                # 复制文件
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
    
    return scene

def main():
    # 获取所有场景文件夹
    scenes = [d for d in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, d))]
    
    print(f"Found {len(scenes)} scenes to process")
    print(f"Each scene will be divided into 50 shards of 100 samples each")
    
    # 使用进程池处理（最多10个进程）
    results = []
    with multiprocessing.Pool(processes=10) as pool:
        # 使用tqdm显示进度条
        with tqdm(total=len(scenes), desc="Processing scenes") as pbar:
            for result in pool.imap_unordered(process_scene, scenes):
                if result:
                    results.append(result)
                pbar.update(1)
    
    print("\nProcessing completed!")
    print(f"Successfully processed {len(results)} scenes")
    print(f"Copied samples to {50 * len(results)} shard directories")

if __name__ == "__main__":
    main()