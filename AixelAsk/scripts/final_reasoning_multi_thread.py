import json
import re
import os
from tqdm import tqdm
import sys
from processing_format import get_row_description, get_col_description, get_row_flattened
from generate_solution_plan import get_solution_plan
from utils.request_gpt import request_gpt_chat, request_gpt_embedding
from utils.processing import clean_table, index_table
from get_sub_table import retrieve_final_subtable, retrieve_final_subtable_add
from concurrent.futures import ThreadPoolExecutor, as_completed
from generate_answer import generate_final_answer, generate_noplan_answer


def process_single_table(index, d, row_prompt, col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt):
    """Process a single table and return the result record."""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]
    answer = ", ".join(item["answer"])

    cleaned_table = clean_table(table)
    indexed_cleaned_table = index_table(cleaned_table)

    # # Example: clean the header (kept for reference)
    # header = table[0]
    # cleaned_header = clean_header(header)
    # cleaned_table = [cleaned_header] + table[1:]
    # cleaned_indexing_table = add_row_index_column(cleaned_table)

    try:
        # Generate natural-language descriptions for rows and columns
        # row_descriptions = get_row_description(cleaned_table, row_prompt)
        row_descriptions = get_row_flattened(cleaned_table)
        col_descriptions = get_col_description(cleaned_table, col_prompt)

        # Generate the solution plan
        solution_plan = get_solution_plan(cleaned_table, question, plan_prompt)

        # If the plan is invalid or has only a single reasoning stage
        if solution_plan is None or len(solution_plan) == 1:
            # Perform reasoning without a plan
            final_answer = generate_noplan_answer(question, indexed_cleaned_table, noplan_reasoning_prompt)
            is_correct = final_answer.lower() == answer.lower()
            record_data = {
                "index": index,
                "question": question,
                "gold_answer": answer,
                "pred_answer": final_answer,
                "is_correct": is_correct,
                "type": "Single stage reasoning or invalid plan",
                "solution_plan": solution_plan,
                "table_text": indexed_cleaned_table,
                "prompt": noplan_reasoning_prompt,
            }
        else:
            # If there are multiple valid stages, perform retrieval
            final_subtable, final_row_indices, final_col_indices = retrieve_final_subtable_add(
                solution_plan, indexed_cleaned_table, row_descriptions, col_descriptions, request_gpt_embedding, question
            )
            final_answer = generate_final_answer(question, solution_plan, final_subtable, final_reasoning_prompt)
            final_answer = final_answer.strip()

            # Check correctness
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
        print(f"Error encountered at index {index}: {e}. Skipping this iteration.")
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

    # Read existing result file and collect already processed indices
    result_file_path = "result/final_reasoning/wtq_plus/gpt3.5/result_1117_wtq+_valid_test_gpt3_5.jsonl"
    existing_indices = set()

    if os.path.exists(result_file_path):
        with open(result_file_path, 'r', encoding='utf-8') as result_file:
            for line in result_file:
                result_data = json.loads(line)
                existing_indices.add(result_data["index"])
    else:
        print(f"{result_file_path} does not exist; skipping processed-index check.")

    true_count = 0
    pass_count = 0

    # Use a thread pool to process each table concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                process_single_table,
                index, d,
                row_prompt, col_prompt, plan_prompt,
                final_reasoning_prompt, noplan_reasoning_prompt
            )
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
