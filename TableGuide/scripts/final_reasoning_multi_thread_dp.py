import json
import re
from tqdm import tqdm
import sys
sys.path.append('/data/yangyuxin/LargeTableRAG')
from processing_format import get_row_description, get_col_description
from generate_solution_plan import get_solution_plan
from utils.request_gpt import request_gpt_chat, request_gpt_embedding
from get_sub_table import retrieve_final_subtable
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.processing import clean_table, index_table
from generate_answer import generate_final_answer, generate_noplan_answer


def process_single_table(index, d, row_prompt, col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt):
    """处理单个表格并返回结果数据"""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]
    answer = ", ".join(item["answer"])

    cleaned_table = clean_table(table)
    indexed_cleaned_table = index_table(cleaned_table)

    # # 清理表头
    # header = table[0]
    # cleaned_header = clean_header(header)
    # cleaned_table = [cleaned_header] + table[1:]
    # cleaned_indexing_table = add_row_index_column(cleaned_table)

    try:
        final_answer = generate_noplan_answer(question, indexed_cleaned_table, noplan_reasoning_prompt)
        is_correct = final_answer.lower() == answer.lower()
        record_data = {
            "index": index,
            "question": question,
            "gold_answer": answer,
            "pred_answer": final_answer,
            "is_correct": is_correct,
            "table_text": table,
            "prompt": noplan_reasoning_prompt,
        }
    except:
        record_data = {
            "index": index,
            "question": question,
            "gold_answer": answer,
            "error": "Overlength. This model's maximum context length is 16385 tokens.",
            "table_text": table,
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

    with open("prompt/get_solution_plan_v2.md", "r") as f:
        plan_prompt = f.read()

    with open("prompt/final_reasoning.md", "r") as f:
        final_reasoning_prompt = f.read()

    with open("prompt/noplan_reasoning.md", "r") as f:
        noplan_reasoning_prompt = f.read()

    true_count = 0
    pass_count = 0

    # 线程池 并发处理每个表格数据
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_single_table, index, d, row_prompt, col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt)
            for index, d in enumerate(data)
        ]

        with open("result/final_reasoning/wtq_plus/gpt4o/result_1117_wtq+_gpt4omini_dp.jsonl", 'a', encoding='utf-8') as f: 
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