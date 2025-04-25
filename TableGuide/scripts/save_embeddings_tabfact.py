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
    embeddings = [request_fn(desc) for desc in tqdm(descriptions, desc="Generating Embeddings")]
    return embeddings


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
        table_id = item.get("table_id")  # ✅ 使用已有 table_id

        if not table_id:
            return None  # 如果缺失 table_id，跳过

        with lock:
            if table_id in seen_table_ids:
                return None
            seen_table_ids.add(table_id)

        cleaned_table = clean_table(table)
        row_descriptions = get_row_flattened(cleaned_table)
        col_descriptions = get_col_description(cleaned_table, col_prompt)

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
        return None


def main():
    input_path = "dataset/tabfact/large_tabfact_test_data_str.jsonl"
    col_prompt_path = "prompt/get_col_template.md"
    output_path = "cache/table_embeddings_tabfact.jsonl"

    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    with open(col_prompt_path, 'r', encoding='utf-8') as f:
        col_prompt = f.read()

    seen_table_ids = load_existing_table_ids(output_path)
    print(f"🔄 Loaded {len(seen_table_ids)} existing table_ids.")

    lock = threading.Lock()

    with open(output_path, 'a', encoding='utf-8') as fout:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(save_embeddings, idx, line, col_prompt, seen_table_ids, lock)
                for idx, line in enumerate(data)
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Saving table embeddings"):
                result = future.result()
                if result and result["table_id"]:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✅ All new table embeddings appended to {output_path}")


if __name__ == "__main__":
    main()
