import os
import json
import pandas as pd
from tqdm import tqdm

def process_shard(shard_path):
    """
    处理单个分片文件夹，生成各split目录下的max_db.csv文件
    :param shard_path: 分片路径，如 '/remote-home/ums_wangdantong/anavi/scratch/vdj/ss/anp_shard-7_samples-100'
    """
    # 获取分片名称
    shard_name = os.path.basename(shard_path.rstrip('/'))
    
    # 遍历所有split目录
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = os.path.join(shard_path, split)
        
        # 检查split目录是否存在
        if not os.path.exists(split_path):
            continue
        
        all_data = []
        
        # 遍历所有场景文件夹
        for scene in tqdm(os.listdir(split_path), desc=f"Processing {shard_name}/{split}"):
            scene_path = os.path.join(split_path, scene)
            json_path = os.path.join(scene_path, "max_db.json")
            
            # 检查max_db.json是否存在
            if not os.path.exists(json_path):
                continue
            
            # 读取JSON数据
            with open(json_path, 'r') as f:
                scene_data = json.load(f)
            
            # 解析数据条目
            for composite_key, max_db_list in scene_data.items():
                try:
                    # 分割键值 (格式: "split-scene-idx")
                    key_parts = composite_key.split('-')
                    if len(key_parts) != 3:
                        raise ValueError(f"Invalid key format: {composite_key}")
                    
                    key_split = key_parts[0]
                    key_scene = key_parts[1]
                    key_idx = key_parts[2]
                    
                    # 验证路径一致性
                    assert key_split == split, f"Split mismatch: {key_split} vs {split}"
                    assert key_scene == scene, f"Scene mismatch: {key_scene} vs {scene}"
                    
                    # 提取数值
                    max_db = float(max_db_list[0])
                    idx = int(key_idx)
                    
                    all_data.append({
                        'shard': shard_name,
                        'split': split,
                        'scene': scene,
                        'idx': idx,
                        'max_db': max_db
                    })
                    
                except Exception as e:
                    print(f"Error processing {composite_key}: {str(e)}")
                    continue
        
        # 生成CSV文件
        if all_data:
            df = pd.DataFrame(all_data)
            output_csv = os.path.join(split_path, "max_db.csv")
            df.to_csv(output_csv, index=False)
            print(f"Saved {len(df)} records to {output_csv}")

def process_all_shards(base_dir):
    """
    处理所有分片文件夹
    :param base_dir: 分片存储的根目录，如 '/remote-home/ums_wangdantong/anavi/scratch/vdj/ss/'
    """
    # 获取所有分片文件夹
    shard_folders = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("anp_shard-") and d.endswith("_samples-100")
    ]
    
    # 处理每个分片
    for shard_path in tqdm(shard_folders, desc="Processing shards"):
        process_shard(shard_path)

if __name__ == "__main__":
    # 配置根目录路径
    BASE_DIR = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss/"
    
    # 执行处理
    process_all_shards(BASE_DIR)
    print("All processing completed!")