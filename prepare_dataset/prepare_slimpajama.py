# from datasets import load_dataset
# import os
# import glob

# slimpajama_sets = {
#     "validation": "validation/chunk*/*",
#     "train": "train/chunk*/*",
#     "test": "test/chunk*/*",
# }
# percentage = 1.0

# for split in slimpajama_sets.keys():
#     source_path = "/work/SlimPajama-627B/datasets--cerebras--SlimPajama-627B/snapshots/2d0accdd58c5d5511943ca1f5ff0e3eb5e293543"
#     filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
#     filenames = filenames[:int(len(filenames) * percentage)]

#     processes = []
#     data = load_dataset("json", data_files=filenames)
#     data['train'].to_json(f"/work/data/json_data/SlimPajama_{split}_data.json", lines=True)
from datasets import load_dataset
import os
import glob

# 定义数据集的分割结构
slimpajama_sets = {
    "validation": "validation/chunk*",
    "train": "train/chunk*",
    "test": "test/chunk*",
}
percentage = 1.0  # 控制处理的 chunk 数量比例（1.0 表示全部）

# 输出目录
output_dir = "/ssd_1234/haozeng/data/json_data"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个数据集分割
for split in slimpajama_sets.keys():
    source_path = "/ssd_1234/haozeng/SlimPajama-627B/datasets--cerebras--SlimPajama-627B/snapshots/2d0accdd58c5d5511943ca1f5ff0e3eb5e293543"
    pattern = slimpajama_sets[split]

    # 获取所有 chunk 目录路径
    chunk_dirs = glob.glob(os.path.join(source_path, pattern), recursive=True)
    chunk_dirs = chunk_dirs[:int(len(chunk_dirs) * percentage)]  # 按比例选取

    # 逐个处理每个 chunk 目录
    for chunk_dir in chunk_dirs:
        # 获取当前 chunk 下的所有数据文件
        data_files = glob.glob(os.path.join(chunk_dir, "*.jsonl.zst"))


        # 构造输出文件名
        chunk_name = os.path.basename(chunk_dir)  # 如 "chunk000"
        output_filename = os.path.join(
            output_dir,
            f"SlimPajama_{split}_{chunk_name}.json"
        )
        # 检查输出文件是否已存在
        if os.path.exists(output_filename):
            print(f"文件 {output_filename} 已存在，跳过处理 {chunk_dir}")
            continue

        # 加载该 chunk 的所有数据
        dataset = load_dataset("json", data_files=data_files)

        # 保存为 JSON 文件
        dataset['train'].to_json(output_filename, lines=True)