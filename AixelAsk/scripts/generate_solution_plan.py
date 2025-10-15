import json
import random
import sys
import re
from tqdm import tqdm
from utils.request_gpt import request_gpt_chat
from utils.processing import list_to_markdown, sample_table_rows
from concurrent.futures import ThreadPoolExecutor, as_completed


def clean_header(header):
    """Clean column names to contain only letters, digits, and underscores. 
    If a column name is empty, return 'null'."""
    cleaned_header = []
    for column_name in header:
        if not column_name.strip():  # Check for empty or whitespace-only column name
            cleaned_name = 'null'
        else:
            # Replace non-alphanumeric/underscore characters
            cleaned_name = re.sub(r'\W+', '_', column_name)
            # Collapse multiple underscores and strip leading/trailing underscores
            cleaned_name = re.sub(r'_+', '_', cleaned_name).strip('_')
        cleaned_header.append(cleaned_name)
    return cleaned_header


def get_solution_plan(table, question, plan_prompt):
    """Generate a multi-stage solution plan using GPT based on the input table and question."""
    header, sampled_rows = sample_table_rows(table)
    markdown_table = list_to_markdown(header, sampled_rows)
    input_text = plan_prompt.format(question=question, table=markdown_table)

    # Set the maximum number of retries
    max_attempts = 10
    for attempt in range(max_attempts):
        solution_plan = request_gpt_chat(input_text)
        
        # Validate the generated solution plan
        plan_dict = validate_solution_plan(solution_plan)
        if plan_dict:
            return plan_dict
        else:
            print(f"Attempt {attempt + 1}: Generated solution plan does not match expected format, retrying...")
    
    raise ValueError("Failed to generate a valid solution plan after multiple attempts.")


def validate_solution_plan(solution_plan):
    """Validate and parse the generated solution plan into a list of dictionaries."""
    try:
        plan_dict = json.loads(solution_plan)
        
        # Check format: must be a list of dicts with Stage, Sub-Level-Question, Action, Top k
        if isinstance(plan_dict, list) and all(
            isinstance(stage, dict) and
            'Stage' in stage and
            'Sub-Level-Question' in stage and
            'Action' in stage and
            'Top k' in stage
            for stage in plan_dict
        ):
            # Case: single-stage plan
            if len(plan_dict) == 1:
                return plan_dict if plan_dict[0]['Action'] == 'Reasoning' else None
            
            # Multi-stage plan validation
            for stage in plan_dict[:-1]:  # Check all but the last stage
                if stage['Action'] != 'Retrieval':
                    return None
            
            # Final stage must be Reasoning
            if plan_dict[-1]['Action'] == 'Reasoning':
                return plan_dict

    except json.JSONDecodeError:
        pass
    return None


def process_single_table(index, d, plan_prompt):
    """Process a single table and return the result record."""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]

    # Clean table header
    header = table[0]
    cleaned_header = clean_header(header)
    cleaned_table = [cleaned_header] + table[1:]

    try:
        # Generate solution plan
        solution_plan = get_solution_plan(cleaned_table, question, plan_prompt)
        record_data = {
            "question": question,
            "solution_plan": solution_plan,
            "table": cleaned_table,
        }
    except Exception as e:
        print(f"Error encountered at index {index}: {e}. Skipping this iteration.")
        record_data = {
            "question": question,
            "solution_plan": "null",
            "table": cleaned_table,
        }

    return record_data


def main():
    with open("dataset/4096.jsonl", 'r') as f:
        data = f.readlines()

    with open("prompt/get_solution_plan_with_topk_v3dot5.md", "r") as f:
        plan_prompt = f.read()

    # Thread pool: process each table concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_single_table, index, d, plan_prompt)
            for index, d in enumerate(data)
        ]

        with open("result/solution_plan/solution_plan_with_topk_1107_v3dot5.jsonl", 'a', encoding='utf-8') as f:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
                result = future.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
