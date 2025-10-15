import json
import random
import sys
import re
from tqdm import tqdm
from utils.request_gpt import request_gpt_chat
from utils.processing import list_to_markdown, sample_table_rows
from concurrent.futures import ThreadPoolExecutor, as_completed


def clean_header(header):
    """清理列名，确保每个列名只包含字母、数字和下划线。如果列名为空，则返回 'null'。"""
    cleaned_header = []
    for column_name in header:
        if not column_name.strip():  # 检查是否为空或仅包含空白字符
            cleaned_name = 'null'
        else:
            # 替换非字母、数字和下划线的字符
            cleaned_name = re.sub(r'\W+', '_', column_name)
            # 确保没有连续的下划线，并去除开头和结尾的下划线
            cleaned_name = re.sub(r'_+', '_', cleaned_name).strip('_')
        
        cleaned_header.append(cleaned_name)
    return cleaned_header 


def get_solution_plan(table, question, plan_prompt):
    header, sampled_rows = sample_table_rows(table)

    markdown_table = list_to_markdown(header, sampled_rows)
    input = plan_prompt.format(question=question, table=markdown_table)
    # print(input)
    # print("****************")

    # 设定最大重试次数
    max_attempts = 10
    for attempt in range(max_attempts):
        solution_plan = request_gpt_chat(input)
        
        # 使用 validate_solution_plan 函数验证 solution_plan
        plan_dict = validate_solution_plan(solution_plan)
        if plan_dict:
            return plan_dict
        else:
            print(f"Attempt {attempt + 1}: Generated solution plan does not match the expected format, retrying...")
    
    raise ValueError("Failed to generate solution plan in the expected format after multiple attempts.")


def validate_solution_plan(solution_plan):
    """验证生成的 solution_plan 是否符合要求并解析为字典列表。"""
    try:
        plan_dict = json.loads(solution_plan)
        
        # 检查解析后的 plan 是否为列表，并包含 Stage、Sub-Level-Question、Action 三个字段
        if isinstance(plan_dict, list) and all(
            isinstance(stage, dict) and 
            'Stage' in stage and 
            'Sub-Level-Question' in stage and 
            'Action' in stage and
            'Top k' in stage for stage in plan_dict
        ):
            # 方案只有一个 stage 的情况
            if len(plan_dict) == 1:
                return plan_dict if plan_dict[0]['Action'] == 'Reasoning' else None
            
            # 方案有多个 stage 的情况
            for stage in plan_dict[:-1]:  # 检查前面的 stages
                if stage['Action'] != 'Retrieval':
                    return None
            
            # 检查最后一个 stage 是否为 "Reasoning"
            if plan_dict[-1]['Action'] == 'Reasoning':
                return plan_dict

    except json.JSONDecodeError:
        pass 
    return None


def process_single_table(index, d, plan_prompt):
    """处理单个表格并返回结果数据"""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]

    # 清理表头
    header = table[0]
    cleaned_header = clean_header(header)
    cleaned_table = [cleaned_header] + table[1:]

    try:
        # 生成 solution plan
        solution_plan = get_solution_plan(cleaned_table, question, plan_prompt)
        record_data = {
            "question": question,
            "solution_plan": solution_plan,
            "table" : cleaned_table,
        }
    except Exception as e:
        print(f"Error encountered {index}: {e}. Skipping this iteration.")
        record_data = {
            "question": question,
            "solution_plan": "null",
            "table" : cleaned_table,
        }

    return record_data


def main():
    with open("dataset/4096.jsonl", 'r') as f:
        data = f.readlines()

    with open("prompt/get_solution_plan_with_topk_v3dot5.md", "r") as f:
        plan_prompt = f.read()

    # 线程池 并发处理每个表格数据
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
