import boto3
import gzip
import glob
import os
import json
from datasets import load_dataset
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

num_proc = 16
batch_size = 10000  # 每批处理 1 万条数据
output_file = "/root/data/json_data/smollm_corpus_train_python-edu.json"
temp_dir = "/root/data/json_data/temp_python_edu"

# 创建临时目录
os.makedirs(temp_dir, exist_ok=True)

# Software Heritage S3 是公开存储桶，使用匿名访问
s3 = boto3.client("s3", region_name="eu-west-1", config=Config(signature_version=UNSIGNED))
bucket_name = "softwareheritage"

# 本地 parquet 文件路径
parquet_files = glob.glob("/root/data/smollm-corpus/python-edu/*.parquet")

def download_contents(blob_id):
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return {"text": "", "download_success": False}

# 从本地 parquet 文件加载
print(f"Loading dataset from {len(parquet_files)} parquet files...")
ds = load_dataset("parquet", data_files=parquet_files, split="train")
total_records = len(ds)
print(f"Total records: {total_records}")

# 分批处理
for i in range(0, total_records, batch_size):
    batch_num = i // batch_size
    temp_file = os.path.join(temp_dir, f"batch_{batch_num:04d}.json")
    
    # 检查该批次是否已处理
    if os.path.exists(temp_file):
        print(f"Batch {batch_num} already processed, skipping...")
        continue
    
    end = min(i + batch_size, total_records)
    print(f"\nProcessing batch {batch_num}: records {i} to {end} ({end-i} records)")
    
    # 处理当前批次
    batch_ds = ds.select(range(i, end))
    batch_ds = batch_ds.map(download_contents, input_columns="blob_id", num_proc=num_proc)
    
    # 过滤并保存
    batch_ds = batch_ds.filter(lambda x: x['download_success'])
    batch_ds.to_json(temp_file, lines=True)
    print(f"Saved batch {batch_num} to {temp_file} ({len(batch_ds)} successful downloads)")

# 合并所有批次文件
print("\nMerging all batch files...")
with open(output_file, 'w') as outf:
    batch_files = sorted(glob.glob(os.path.join(temp_dir, "batch_*.json")))
    for batch_file in batch_files:
        print(f"Merging {batch_file}...")
        with open(batch_file, 'r') as inf:
            for line in inf:
                outf.write(line)

print(f"\nAll done! Final file saved to: {output_file}")
print(f"Temporary files in: {temp_dir}")
print(f"You can delete temp files with: rm -rf {temp_dir}")