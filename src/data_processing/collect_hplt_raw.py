"""
Collect raw HPLT JSONL files into one JSONL file.

Example:
    python -m src.data_processing.collect_hplt_raw --input_dir data/hplt_v3/raw --output_file data/hplt_v3/extracted_raw/all_data.jsonl
"""

import os
import glob
import gzip
import json
import argparse
import io
from pathlib import Path
from tqdm import tqdm
import zstandard as zstd # type: ignore

def process_raw_data(input_dir, output_file):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extensions = ['*.jsonl', '*.jsonl.gz', '*.jsonl.zst']
    files = []
    for ext in extensions:
        files.extend(input_path.rglob(ext))
    
    files = sorted(files)
    
    if not files:
        print(f"No files found in {input_dir}")
        return

    print(f"Found {len(files)} files to process.")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in tqdm(files, desc="Processing files"):
            try:
                if str(file_path).endswith('.zst'):
                    dctx = zstd.ZstdDecompressor()
                    with open(file_path, 'rb') as fh:
                        with dctx.stream_reader(fh) as reader:
                             text_reader = io.TextIOWrapper(reader, encoding='utf-8')
                             for line in text_reader:
                                 if line.strip():
                                     out_f.write(line)
                else:
                    handler = gzip.open(file_path, 'rt', encoding='utf-8') if str(file_path).endswith('.gz') else open(file_path, 'r', encoding='utf-8')
                    with handler as in_f:
                        for line in in_f:
                            if line.strip():
                                out_f.write(line)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and concatenate raw HPLT data.")
    parser.add_argument("--input_dir", type=str, default="data/hplt_v3/raw", help="Directory containing raw data.")
    parser.add_argument("--output_file", type=str, default="data/hplt_v3/cleaned/all_data.jsonl", help="Output file path.")
    
    args = parser.parse_args()
    
    process_raw_data(args.input_dir, args.output_file)
