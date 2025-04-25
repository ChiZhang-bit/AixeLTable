import json
import re
import sys
sys.path.append('/data/yangyuxin/LargeTableRAG')
from processing_format import get_row_description, get_col_description, get_row_flattened
from generate_solution_plan import get_solution_plan
from utils.request_gpt import request_gpt_chat, request_gpt_embedding
from get_sub_table import retrieve_final_subtable, retrieve_final_subtable_add
from utils.processing import clean_table, index_table
from generate_answer import generate_final_answer, generate_noplan_answer


def process_single_table(index, d, row_prompt, col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt):
    """处理单个表格并返回结果数据"""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]
    answer = ", ".join(item["answer"]).lower()
    print(len(table))

    cleaned_table = clean_table(table)
    indexed_cleaned_table = index_table(cleaned_table)

    # # 清理表头
    # header = table[0]
    # cleaned_header = clean_header(header)
    # cleaned_table = [cleaned_header] + table[1:]
    # cleaned_indexing_table = add_row_index_column(cleaned_table)

    try:
        # 生成行、列的自然语言描述
        row_descriptions = get_row_flattened(cleaned_table)
        # row_descriptions = get_row_description(cleaned_table, row_prompt)
        col_descriptions = get_col_description(cleaned_table, col_prompt)

        # 生成 solution plan
        solution_plan = get_solution_plan(cleaned_table, question, plan_prompt)

        # 如果 plan 无效或仅有一个 stage 且为 Reasoning
        if solution_plan is None or len(solution_plan) == 1:
            # 执行无需 plan 的推理生成
            final_answer = generate_noplan_answer(question, indexed_cleaned_table, noplan_reasoning_prompt)
            is_correct = final_answer.lower() == answer.lower()
            record_data = {
                "index": index,
                "question": question,
                "gold_answer": answer,
                "pred_answer": final_answer,
                "is_correct": is_correct,
                "type": "Single stage reasoning or invalid plan",
                "solution_plan":solution_plan,
                "table_text": indexed_cleaned_table,
                "prompt": noplan_reasoning_prompt,
            }
        else:
            # 如果有多个 stage 且经过验证有效，则进行Retrieval
            final_subtable, final_row_indices, final_col_indices = retrieve_final_subtable_add(
                solution_plan, indexed_cleaned_table, row_descriptions, col_descriptions, request_gpt_embedding, question
            )
            final_answer = generate_final_answer(question, solution_plan, final_subtable, final_reasoning_prompt)
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
                "solution_plan": solution_plan,
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
            "table_text": table,
            "prompt": noplan_reasoning_prompt,
        }


def main():
    # 加载输入数据
    with open("dataset/wikitq/valid/4096/4096.jsonl", 'r') as f:
        data = f.readlines()

    # 加载所需的 prompt 文件
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

    for index, d in enumerate(data):

        # 处理单个表格数据
        result = process_single_table(index, d, row_prompt, col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt)

        # 保存结果
        with open("result/single_test_result.json", 'w', encoding='utf-8') as f: 
            json.dump(result, f, ensure_ascii=False, indent=4)

        print("Result saved to result/single_test_result.json")

if __name__ == "__main__":
    main()
