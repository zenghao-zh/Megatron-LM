from datasets import load_dataset
import os
import glob
from collections import defaultdict

percentage = 1.0
split = 'train'
source_path = "/ssd_1234/haozeng/starcoder/data"
output_dir = "/ssd_1234/haozeng/data/json_data"

assert split == "train" #  starcoder only has train data
filenames = glob.glob(os.path.join(source_path, "*/*.parquet"), recursive=True)
# only retrain subsets that follow the prefix in filenames_subset

filenames = filenames[:int(len(filenames) * percentage)]

# 按父目录名归类
groups = defaultdict(list)
for f in filenames:
    parent_dir = os.path.basename(os.path.dirname(f))  # 获取父目录名
    groups[parent_dir].append(f)

for category, files in groups.items():
    print(f"Processing category: {category} with {len(files)} files")
    dataset = load_dataset("parquet", data_files=files)
    output_path = os.path.join(output_dir, f"starcode_{split}_{category}.json")
    dataset['train'].to_json(output_path, lines=True)
    print(f"Saved to: {output_path}")