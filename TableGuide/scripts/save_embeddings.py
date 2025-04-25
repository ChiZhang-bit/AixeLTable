import json
import hashlib
from tqdm import tqdm
import sys
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('/data/yangyuxin/LargeTableRAG')

from processing_format import get_row_flattened, get_col_description
from utils.processing import clean_table
from utils.request_gpt import request_gpt_embedding


def get_embeddings(descriptions, request_fn):
    """获取每个描述的embedding"""
    # return [request_fn(desc) for desc in descriptions]
    embeddings = [request_fn(desc) for desc in tqdm(descriptions, desc="Generating Embeddings")]
    return embeddings


def get_table_id_from_text(table):
    """根据 table_text 内容生成唯一 ID（使用 hash）"""
    table_str = json.dumps(table, sort_keys=True)
    return hashlib.sha1(table_str.encode('utf-8')).hexdigest()


def load_existing_table_ids(output_path):
    """读取已保存的 embedding 文件中的 table_id 集合"""
    existing_ids = set()
    if not os.path.exists(output_path):
        return existing_ids

    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                table_id = item.get("table_id")
                if table_id:
                    existing_ids.add(table_id)
            except:
                continue
    return existing_ids


def save_embeddings(index, line, col_prompt, seen_table_ids, lock):
    """处理单个表格条目，提取并保存表的行列 embedding，如果是重复表则跳过"""
    try:
        item = json.loads(line)
        table = item["table_text"]
        statement = item["statement"]

        # 生成唯一 table_id
        table_id = get_table_id_from_text(table)

        with lock:
            if table_id in seen_table_ids:
                return None
            seen_table_ids.add(table_id)

        # 清洗表格
        cleaned_table = clean_table(table)

        # row_descriptions = 'test'
        # col_descriptions = 'test'

        # row_embeddings = 'testttt'
        # col_embeddings = 'testttt'

        # 获取描述
        row_descriptions = get_row_flattened(cleaned_table)
        col_descriptions = get_col_description(cleaned_table, col_prompt)

        # 获取 embedding
        row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
        col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

        return {
            "table_id": table_id,
            "statement": statement,
            "row_descriptions": row_descriptions,
            "row_embeddings": row_embeddings,
            "col_descriptions": col_descriptions,
            "col_embeddings": col_embeddings,
            "table_text": table
        }
    except:
        return {
            "table_id": None,
            "statement": statement,
            "row_descriptions": None,
            "row_embeddings": None,
            "col_descriptions": None,
            "col_embeddings": None,
            "table_text": table
        }


def process_table_embeddings(input_path, output_path, col_prompt_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    with open(col_prompt_path, 'r', encoding='utf-8') as f:
        col_prompt = f.read()

    seen_table_ids = load_existing_table_ids(output_path)
    print(f"🔄 Loaded {len(seen_table_ids)} existing table_ids from {output_path}.")

    lock = threading.Lock()

    with open(output_path, 'a', encoding='utf-8') as fout:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(save_embeddings, idx, line, col_prompt, seen_table_ids, lock)
                for idx, line in enumerate(data)
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Saving embeddings to {output_path}"):
                result = future.result()
                if result and result["table_id"]:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✅ All new table embeddings appended to {output_path}")


def main():
    col_prompt_path = "prompt/get_col_template.md"

    configs = [
        ("dataset/wikitq/valid/1024-2048/1024-2048_sample.jsonl", "cache/table_embeddings_2k.jsonl"),
        ("dataset/wikitq/valid/2048-3072/2048-3072_sample.jsonl", "cache/table_embeddings_3k.jsonl"),
        ("dataset/wikitq/valid/3072-4096/3072-4096.jsonl", "cache/table_embeddings_4k.jsonl"),
    ]

    for input_path, output_path in configs:
        print(f"🚀 Processing {input_path}...")
        process_table_embeddings(input_path, output_path, col_prompt_path)


if __name__ == "__main__":
    main()
