from utils.request_gpt import request_gpt_chat
import re

def clean_qwen_output(output: str):
    """
    清除 Qwen 模型生成内容中的奇怪前缀，例如 '6user\\n'、'PARTICULARS\\nuser\\n' 等。
    """
    return re.sub(r"^(?:\d*user\\n|PARTICULARS\\nuser\\n|assistant\\n)?", "", output.strip(), flags=re.IGNORECASE)


def generate_final_answer_plan(question, plan, final_subtable_with_header, prompt):

    col_headers = final_subtable_with_header[0]
    subtable = final_subtable_with_header[1:]
    subtable_md = "| " + " | ".join(col_headers) + " |\n"
    subtable_md += "| " + " | ".join(["---"] * len(col_headers)) + " |\n"
    for row in subtable:
        subtable_md += "| " + " | ".join(map(str, row)) + " |\n"

    plan_text = ""
    for stage in plan:
        plan_text += f"Stage {stage['Stage']}:\n"
        plan_text += f"  Sub-Level-Question: {stage['Sub-Level-Question']}\n"

    prompt = prompt.format(question = question, table=subtable_md, plan=plan_text)
    # print(prompt)
    final_answer = request_gpt_chat(prompt=prompt)

    return final_answer


def generate_final_answer_DAG(question, dag, final_subtable_with_header, prompt):

    col_headers = final_subtable_with_header[0]
    subtable = final_subtable_with_header[1:]
    subtable_md = "| " + " | ".join(col_headers) + " |\n"
    subtable_md += "| " + " | ".join(["---"] * len(col_headers)) + " |\n"
    for row in subtable:
        subtable_md += "| " + " | ".join(map(str, row)) + " |\n"

    plan_text = ""
    for stage in dag:
        plan_text += f"Node {stage['NodeID']}:\n"
        plan_text += f"Sub-Level-Question: {stage['Sub-Level-Question']}\n"
        plan_text += f"Next Node: {stage['Next']}\n"

    prompt = prompt.format(question = question, table=subtable_md, dag=plan_text)
    # print(prompt)
    final_answer_raw = request_gpt_chat(prompt=prompt)
    
    # 清洗 Qwen 的输出
    final_answer = clean_qwen_output(final_answer_raw)

    return final_answer


def generate_noplan_answer(question, table_with_header, prompt):

    col_headers = table_with_header[0]
    table = table_with_header[1:]
    table_md = "| " + " | ".join(col_headers) + " |\n"
    table_md += "| " + " | ".join(["---"] * len(col_headers)) + " |\n"
    for row in table:
        table_md += "| " + " | ".join(map(str, row)) + " |\n"

    prompt = prompt.format(question = question, table=table_md)
    # print(prompt)
    final_answer = request_gpt_chat(prompt=prompt)

    return final_answer