import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 源目录路径
src_base = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss/anp_shard-100_samples-5000/train"
# 目标根目录
dst_root = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss"

# 要处理的JSON文件列表
json_files = ['gt.json', 'metadata.json', 'max_db.json']

def process_scene(scene_path):
    scene_name = os.path.basename(scene_path)
    
    # 读取所有JSON文件内容
    json_contents = {}
    for file_name in json_files:
        file_path = os.path.join(scene_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                json_contents[file_name] = json.load(f)
        else:
            print(f"Warning: {file_path} not found for scene {scene_name}")
            json_contents[file_name] = None
    
    # 确定样本数量（根据gt.json）
    if json_contents['gt.json']:
        total_samples = len(json_contents['gt.json'])
    else:
        total_samples = 5000  # 默认值
    
    # 分片处理
    num_shards = 50
    samples_per_shard = total_samples // num_shards
    
    for shard_idx in range(num_shards):
        # 计算当前分片的样本范围
        start_idx = shard_idx * samples_per_shard
        end_idx = (shard_idx + 1) * samples_per_shard
        
        # 计算目标目录的n值
        n = shard_idx + 2
        dest_dir = os.path.join(dst_root, f"anp_shard-{n}_samples-100", "train", scene_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        # 处理每个JSON文件
        for file_name in json_files:
            if json_contents[file_name] is None:
                continue  # 如果文件不存在则跳过
                
            content = json_contents[file_name]
            new_content = {}
            
            # 特殊处理max_db.json - 保留原始键名
            if file_name == 'max_db.json':
                for key, value in content.items():
                    # 提取索引值
                    if '-' in key:
                        # 键名格式：train-场景名-索引
                        parts = key.split('-')
                        try:
                            idx = int(parts[-1])
                        except:
                            continue
                    else:
                        try:
                            idx = int(key)
                        except:
                            continue
                    
                    # 检查是否在当前分片范围内
                    if start_idx <= idx < end_idx:
                        # 直接使用原始键名
                        new_content[key] = value
            
            # 处理其他JSON文件 - 保留原始键名
            else:
                for key, value in content.items():
                    try:
                        idx = int(key)
                    except:
                        continue
                    
                    # 检查是否在当前分片范围内
                    if start_idx <= idx < end_idx:
                        # 直接使用原始键名
                        new_content[key] = value
            
            # 保存处理后的内容
            dest_path = os.path.join(dest_dir, file_name)
            
            # 如果文件已存在则删除重建
            if os.path.exists(dest_path):
                os.remove(dest_path)
                
            with open(dest_path, 'w') as f:
                json.dump(new_content, f, indent=4)
    
    return scene_name

def main():
    # 获取所有场景目录
    scene_dirs = []
    for scene_name in os.listdir(src_base):
        scene_path = os.path.join(src_base, scene_name)
        if os.path.isdir(scene_path):
            scene_dirs.append(scene_path)
    
    print(f"Found {len(scene_dirs)} scenes to process")
    
    # 使用线程池处理（最多5个线程）
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有任务
        futures = [executor.submit(process_scene, scene_dir) for scene_dir in scene_dirs]
        
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing scene: {e}")
    
    print("\nProcessing completed!")
    print(f"Successfully processed {len(results)} scenes")

if __name__ == "__main__":
    main()