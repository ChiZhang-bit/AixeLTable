import random
import re
import copy

def sample_table_rows(table, num_samples=5):
    """
    从二维数组的表格中随机抽取指定数量(num_samples)的行。
    """
    # 获取表头
    header = table[0]
    
    # 抽取非表头部分的指定数量的行
    rows = random.sample(table[1:], num_samples)
    
    return header, rows


def list_to_markdown(header, rows):
    markdown_table = "| " + " | ".join(header) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        # row = str(row)
        # print(row)
        markdown_table += "| " + " | ".join(row) + " |\n"
    return markdown_table


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


def clean_table(table):
    # 深拷贝表格，避免修改原始表格
    table_copy = copy.deepcopy(table)
    # 清理表头
    header = table_copy[0]
    cleaned_header = clean_header(header)
    cleaned_table = [cleaned_header] + table_copy[1:]
    
    return cleaned_table


def index_table(table):
    # 深拷贝表格，避免修改原始表格
    table_copy = copy.deepcopy(table)
    # 添加 "row index" 列名
    table_copy[0].insert(0, "row index")
    
    # 对每一行数据，添加从1开始的行索引
    for i in range(1, len(table_copy)):
        table_copy[i].insert(0, f"row {i}")
    
    return table_copy