import os
import json
import pandas as pd
from tqdm import tqdm
import re

def consolidate_maxdb_json(shard_folders, output_csv="consolidated_max_db.csv"):
    """
    功能：整合多个分片的 max_db.json 到单个 CSV
    参数：
      shard_folders: 分片文件夹路径列表（如 ["/path/to/anp_shard-2_samples-100"]）
      output_csv: 输出 CSV 的完整路径（默认保存到当前目录）
    """
    all_data = []
    
    for shard_folder in tqdm(shard_folders, desc="Processing shards"):
        # === 分片编号过滤 ===
        shard_name = os.path.basename(shard_folder.rstrip('/'))
        match = re.match(r'anp_shard-(\d+)_samples-100', shard_name)
        if not match:
            print(f"无效分片格式: {shard_name}, 跳过")
            continue
            
        shard_num = int(match.group(1))
        if not (2 <= shard_num <= 39):
            print(f"跳过编号 {shard_num} 的分片")
            continue
        
        # === 遍历目录处理 JSON ===
        for root, dirs, files in os.walk(shard_folder):
            if "max_db.json" not in files:
                continue
                
            json_path = os.path.join(root, "max_db.json")
            split = os.path.basename(os.path.dirname(root))  # 提取 split 名称 (train/val/test)
            scene = os.path.basename(root)                   # 提取场景名称
            
            with open(json_path, 'r') as f:
                scene_data = json.load(f)
            
            # === 解析 JSON 条目 ===
            for composite_key, max_db_list in scene_data.items():
                try:
                    key_parts = composite_key.split('-')
                    split_from_key = key_parts[0]
                    scene_from_key = key_parts[1]
                    idx = key_parts[2]
                    
                    # 验证路径解析和键的一致性
                    assert split == split_from_key, f"Split名称不一致: {split} vs {split_from_key}"
                    assert scene == scene_from_key, f"Scene名称不一致: {scene} vs {scene_from_key}"
                    
                    # 转换为数值类型
                    max_db_val = float(max_db_list[0])
                    idx = int(idx)
                    
                    all_data.append({
                        'shard': shard_name,
                        'split': split,
                        'scene': scene,
                        'idx': idx,
                        'max_db': max_db_val
                    })
                except Exception as e:
                    print(f"解析错误 {composite_key}: {str(e)}")
                    continue

    # === 生成 CSV 并返回 ===
    df = pd.DataFrame(all_data)
    df = df[['shard', 'split', 'scene', 'idx', 'max_db']]  # 确保列顺序
    
    # 验证数据完整性
    print("\n[状态报告]")
    print(f"总样本数: {len(df)}")
    print(f"输出路径: {os.path.abspath(output_csv)}")
    
    df.to_csv(output_csv, index=False)
    return df

# 示例调用（生成到指定路径）
if __name__ == "__main__":
    # === 配置参数 ===
    BASE_DIR = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss/"  # 根据实际修改
    OUTPUT_CSV = "/remote-home/ums_wangdantong/anavi/scratch/vdj/ss/max_db.csv"  # 自定义输出路径
    
    # === 自动获取分片路径 ===
    all_shards = [
        os.path.join(BASE_DIR, d) 
        for d in os.listdir(BASE_DIR) 
        if re.match(r'anp_shard-\d+_samples-100', d)
    ]
    
    # === 执行合并 ===
    consolidate_maxdb_json(
        shard_folders=all_shards,
        output_csv=OUTPUT_CSV  # 指定输出路径
    )