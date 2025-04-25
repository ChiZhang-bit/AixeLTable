import json
import random
import sys
import re
from tqdm import tqdm
sys.path.append('/data/yangyuxin/LargeTableRAG')
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



def load_question_type_map(label_file):
    """读取标注文件并返回 question 到 type 的映射字典"""
    question_type_map = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question_type_map[item["question"]] = item["type"]
    return question_type_map


def get_dag(table, question, question_type, dag_prompt_template):

    header, sampled_rows = sample_table_rows(table)
    markdown_table = list_to_markdown(header, sampled_rows)
    # input = dag_prompt.format(question=question, table=markdown_table)
    # print(input)
    # print("****************")

    # 根据 question_type 读取对应的 few-shot 示例内容
    few_shot_file_map = {
        'parallel': 'prompt/fewshot/fewshot_parallel.txt',
        'sequential': 'prompt/fewshot/fewshot_sequential.txt',
        'hybrid': 'prompt/fewshot/fewshot_hybrid.txt',
        'tabfact': 'prompt/fewshot/fewshot_tabfact.txt'
    }

    if question_type not in few_shot_file_map:
        raise ValueError(f"Unsupported question_type: {question_type}")

    fewshot_path = few_shot_file_map[question_type]

    with open(fewshot_path, 'r', encoding='utf-8') as f:
        fewshot_examples = f.read()

    dag_prompt = dag_prompt_template.format(
        fewshot=fewshot_examples.strip(),
        question=question,
        table=markdown_table
    )

    # 设定最大重试次数
    max_attempts = 10
    for attempt in range(max_attempts):
        dag = request_gpt_chat(dag_prompt)
        # print(dag)
        
        # 使用 validate_dag 函数验证 dag
        dag_dict = validate_dag(dag)
        if dag_dict:
            return dag_dict
        else:
            print(f"Attempt {attempt + 1}: Generated DAG does not match the expected format, retrying...")
    
    raise ValueError("Failed to generate DAG in the expected format after multiple attempts.")


def validate_dag(dag_plan):
    """
    验证生成的 DAG plan 是否符合要求：
    1. 是合法的 JSON 格式；
    2. 每个节点包含 NodeID, Sub-Level-Question, Action, Top k, Next 五个字段；
    3. 所有节点的 Action 必须是 "Retrieval" 或 "Reasoning"；
    4. 最终节点（没有后继节点）的 Action 必须是 "Reasoning"；
    5. 不存在环结构，即图是有向无环的（DAG）。
    """
    try:
        dag_plan = dag_plan.strip().strip("```").strip()
        match = re.search(r'(\[\s*\{.*\}\s*\])', dag_plan, re.DOTALL)
        if match:
            dag = json.loads(match.group(1))
        else:
            return None

        # dag = json.loads(dag_plan)
        
        # 检查 DAG 是否为节点列表，且每个节点包含必备字段
        required_fields = {'NodeID', 'Sub-Level-Question', 'Action', 'Top k', 'Next'}
        if not (isinstance(dag, list) and all(isinstance(node, dict) and required_fields.issubset(node.keys()) for node in dag)):
            return None

        node_dict = {node['NodeID']: node for node in dag}

        # Action 字段检查
        if not all(node['Action'] in ['Retrieval', 'Reasoning'] for node in dag):
            return None

        # 最终节点（Next为空）必须是 Reasoning
        for node in dag:
            if len(node['Next']) == 0 and node['Action'] != 'Reasoning':
                return None

        # 检查是否有环（拓扑排序）
        visited = set()
        on_stack = set()

        def has_cycle(node_id):
            if node_id in on_stack:
                return True  # 有环
            if node_id in visited:
                return False  # 已访问且无环

            visited.add(node_id)
            on_stack.add(node_id)
            for next_id in node_dict[node_id]['Next']:
                if next_id not in node_dict or has_cycle(next_id):
                    return True
            on_stack.remove(node_id)
            return False

        if any(has_cycle(node['NodeID']) for node in dag):
            return None

        return dag

    except json.JSONDecodeError:
        pass

    return None


def process_single_table(index, d, dag_prompt):
    """处理单个表格并返回结果数据"""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]

    # 清理表头
    header = table[0]
    cleaned_header = clean_header(header)
    cleaned_table = [cleaned_header] + table[1:]

    question_type_map = load_question_type_map("dataset/wikitq/valid/4096/4096_daglabeled.jsonl")
    question_type = question_type_map.get(question, "hybrid")  # 默认用 hybrid fallback

    try:
        # 生成 solution plan
        dag = get_dag(cleaned_table, question, question_type, dag_prompt)
        record_data = {
            "question": question,
            "question_type":question_type,
            "DAG": dag,
            "table" : cleaned_table,
        }
    except Exception as e:
        print(f"Error encountered {index}: {e}. Skipping this iteration.")
        record_data = {
            "question": question,
            "DAG": "null",
            "table" : cleaned_table,
        }

    return record_data


def main():
    with open("dataset/wikitq/valid/4096/4096_small.jsonl", 'r') as f:
        data = f.readlines()

    with open("prompt/get_dag_notype.md", "r") as f:
        dag_prompt = f.read()

    # 线程池 并发处理每个表格数据
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(process_single_table, index, d, dag_prompt)
            for index, d in enumerate(data)
        ]

        with open("new_result/dag/generated_dag.jsonl", 'a', encoding='utf-8') as f: 
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
                result = future.result()
                
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                

if __name__ == "__main__":
    main()
