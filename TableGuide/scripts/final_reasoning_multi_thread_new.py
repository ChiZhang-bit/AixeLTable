import json
import re
import os
from tqdm import tqdm
import sys
sys.path.append('/data/yangyuxin/LargeTableRAG')
from processing_format import get_row_description, get_col_description, get_row_flattened
from generate_dag import get_dag
from utils.request_gpt import request_gpt_chat, request_gpt_embedding
from utils.processing import clean_table, index_table
from get_sub_table import retrieve_final_subtable, retrieve_final_subtable_DAG
from concurrent.futures import ThreadPoolExecutor, as_completed
from generate_answer import generate_final_answer, generate_noplan_answer


def load_question_type_map(label_file):
    """读取标注文件并返回 question 到 type 的映射字典"""
    question_type_map = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question_type_map[item["question"]] = item["type"]
    return question_type_map


def process_single_table(index, d, row_prompt, col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt, question_type_map):
    """处理单个表格并返回结果数据"""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]
    answer = ", ".join(item["answer"])
    question_type = question_type_map.get(question, "hybrid")  # 默认用 hybrid fallback

    cleaned_table = clean_table(table)
    indexed_cleaned_table = index_table(cleaned_table)

    # # 清理表头
    # header = table[0]
    # cleaned_header = clean_header(header)
    # cleaned_table = [cleaned_header] + table[1:]
    # cleaned_indexing_table = add_row_index_column(cleaned_table)

    try:
        # 生成行、列的自然语言描述
        # row_descriptions = get_row_description(cleaned_table, row_prompt)
        row_descriptions = get_row_flattened(cleaned_table)
        col_descriptions = get_col_description(cleaned_table, col_prompt)

        # 生成 DAG
        dag = get_dag(cleaned_table, question, question_type, plan_prompt)

        # 如果 DAG 无效或仅有一个Node
        if dag is None or len(dag) == 1:
            # 执行无需 plan 的推理生成
            final_answer = generate_noplan_answer(question, indexed_cleaned_table, noplan_reasoning_prompt)
            is_correct = final_answer.lower() == answer.lower()
            record_data = {
                "index": index,
                "question": question,
                "gold_answer": answer,
                "pred_answer": final_answer,
                "is_correct": is_correct,
                "type": "1 node dag or invalid dag",
                "DAG": dag,
                "table_text": indexed_cleaned_table,
                "prompt": noplan_reasoning_prompt,
            }
        else:
            # 如果有多个 stage 且经过验证有效，则进行Retrieval
            final_subtable, final_row_indices, final_col_indices = retrieve_final_subtable_DAG(
                dag, indexed_cleaned_table, row_descriptions, col_descriptions, request_gpt_embedding, question
            )
            final_answer = generate_final_answer(question, dag, final_subtable, final_reasoning_prompt)
            final_answer = final_answer.strip()

            # 检查答案是否正确
            is_correct = final_answer.lower() == answer.lower()
            record_data = {
                "index": index,
                "question": question,
                "gold_answer": answer,
                "pred_answer": final_answer,
                "is_correct": is_correct,
                "type": "Multiple stages",
                "DAG": dag,
                "final_sub_table": final_subtable,
                "final_row_indices": [int(idx) for idx in final_row_indices],
                "final_col_indices": [int(idx) for idx in final_col_indices],
                "row_descriptions": row_descriptions,
                "col_descriptions": col_descriptions,
                "table_text": indexed_cleaned_table,
                "prompt": final_reasoning_prompt,
            }

    except Exception as e:
        print(f"Error encountered {index}: {e}. Skipping this iteration.")
        final_answer = generate_noplan_answer(question, indexed_cleaned_table, noplan_reasoning_prompt)
        is_correct = final_answer.lower() == answer.lower()
        record_data = {
            "index": index,
            "question": question,
            "gold_answer": answer,
            "pred_answer": final_answer,
            "type": "Error generation",
            "is_correct": is_correct,
            "error": str(e),
            "table_text": indexed_cleaned_table,
            "prompt": noplan_reasoning_prompt,
        }

    return record_data


def main():
    with open("dataset/wikitq+/wiki+test_valid.jsonl", 'r') as f:
        data = f.readlines()

    with open("prompt/get_row_template.md", "r") as f:
        row_prompt = f.read()

    with open("prompt/get_col_template.md", "r") as f:
        col_prompt = f.read()

    with open("prompt/get_solution_plan_with_topk_v3dot5.md", "r") as f:
        plan_prompt = f.read()

    with open("prompt/final_reasoning.md", "r") as f:
        final_reasoning_prompt = f.read()

    with open("prompt/noplan_reasoning.md", "r") as f:
        noplan_reasoning_prompt = f.read()

    question_type_map = load_question_type_map("dataset/wikitq/valid/4096/4096_daglabeled.jsonl")

    # 读取已有结果文件，提取已处理的索引

    result_file_path = "result/final_reasoning/wtq_plus/gpt3.5/result_1117_wtq+_valid_test_gpt3_5.jsonl"
    existing_indices = set()

    if os.path.exists(result_file_path):
        with open(result_file_path, 'r', encoding='utf-8') as result_file:
            for line in result_file:
                result_data = json.loads(line)
                existing_indices.add(result_data["index"])
    else:
        print(f"{result_file_path} 文件不存在，将跳过已有结果的检查。")

    true_count = 0
    pass_count = 0

    # 线程池 并发处理每个表格数据
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_single_table, index, d, row_prompt, col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt, question_type_map)
            for index, d in enumerate(data) if index not in existing_indices
        ]

        with open(result_file_path, 'a', encoding='utf-8') as f: 
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
                result = future.result()
                
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                
                if "is_correct" in result and result["is_correct"]:
                    true_count += 1
                if "error" in result:
                    pass_count += 1

    print("True count:", true_count)
    print("Pass count:", pass_count)


if __name__ == "__main__":
    main()