import json
import sys
import re
sys.path.append('/data/yangyuxin/LargeTableRAG')
from utils.request_gpt import request_gpt_chat, request_gpt_embedding
from scripts.processing_format import get_row_description, get_col_description
from scripts.generate_solution_plan import get_solution_plan
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from scripts.schema_linking import rewrite_question


def get_embeddings(descriptions, request_gpt_embedding):
    """获取每个描述的embedding"""
    embeddings = [request_gpt_embedding(desc) for desc in tqdm(descriptions, desc="Generating Embeddings")]
    return embeddings


# def map_descriptions_to_table_indices(row_descriptions, col_descriptions):

#     """构建行列描述与原始表格索引的映射关系."""

#     row_mapping = {desc: i for i, desc in enumerate(row_descriptions)}
#     col_mapping = {desc: j for j, desc in enumerate(col_descriptions)}
#     return row_mapping, col_mapping


def retrieve_rows_by_string_match(table, question):
    """
    根据字符串匹配，在表格中找到与问题相关的行索引。
    """
    # 去除 question 末尾的标点符号并转换为小写
    question_cleaned = re.sub(r'[^\w\s]$', '', question.strip()).lower()  # 保留字母、数字和空格，去除末尾标点
    question_words = set(word for word in re.split(r'\W+', question_cleaned) if word)

    matching_rows = set()

    for row_index, row in enumerate(reversed(table[1:])):
        actual_row_index = len(table) - row_index - 2    # -2是因为还要去掉header行
        for cell in row:
            # 将 cell 内容转换为小写并按非字母数字字符进行分割
            cell_words = set(
                word for word in re.split(r'\W+', str(cell).lower())
                if (len(word) > 3 or word.isdigit()) and word
                ) 

            # 检查 cell_words 和 question_words 是否有交集
            if question_words.intersection(cell_words):
                # print(f"Matched row index: {actual_row_index}, cell content: '{cell}', in question: '{question}'")
                matching_rows.add(actual_row_index)
    return list(matching_rows)


def retrieve_top_relevant_rows_cols(stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk):

    """对每个 retrieval 阶段进行完整的行和列相似度排序，然后选择前5个。"""
    if topk == 'all':
        topk = len(row_embeddings)
    else:
        topk = int(topk)

    sub_level_question = stage['Sub-Level-Question']
    rewrited_sub_level_question = rewrite_question(sub_level_question, header)
    question_embedding = request_gpt_embedding(rewrited_sub_level_question)

    # 计算与行描述的相似度并排序
    row_similarities = cosine_similarity([question_embedding], row_embeddings)[0]
    sorted_row_indices = np.argsort(-row_similarities)  # 按相似度降序排序

    num_rows_to_select = min(topk, len(sorted_row_indices))  # 获取前k行索引
    top_sorted_rows = sorted_row_indices[:num_rows_to_select]

    # 计算与列描述的相似度并排序
    col_similarities = cosine_similarity([question_embedding], col_embeddings)[0]
    sorted_col_indices = np.argsort(-col_similarities)  # 按相似度降序排序

    num_cols_to_select = min(5, len(sorted_col_indices))
    top_sorted_cols = sorted_col_indices[:num_cols_to_select]  # 获取前k列索引

    return top_sorted_rows, top_sorted_cols


def retrieve_top_relevant_rows_cols_notopk(question, row_embeddings, col_embeddings, request_gpt_embedding, header):

    """对每个 retrieval 阶段进行完整的行和列相似度排序，然后选择前5个。"""

    rewrited_sub_level_question = rewrite_question(question, header)
    question_embedding = request_gpt_embedding(rewrited_sub_level_question)

    # 计算与行描述的相似度并排序
    row_similarities = cosine_similarity([question_embedding], row_embeddings)[0]
    sorted_row_indices = np.argsort(-row_similarities)  # 按相似度降序排序

    num_rows_to_select = min(60, len(sorted_row_indices))  # 获取前k行索引
    top_sorted_rows = sorted_row_indices[:num_rows_to_select]

    # 计算与列描述的相似度并排序
    col_similarities = cosine_similarity([question_embedding], col_embeddings)[0]
    sorted_col_indices = np.argsort(-col_similarities)  # 按相似度降序排序

    num_cols_to_select = min(5, len(sorted_col_indices))
    top_sorted_cols = sorted_col_indices[:num_cols_to_select]  # 获取前k列索引

    return top_sorted_rows, top_sorted_cols


def retrieve_final_subtable(solution_plan, indexed_table, row_descriptions, col_descriptions, request_gpt_embedding, question):
    
    """获取所有 retrieval 阶段的前5行列索引，合并后生成最终子表。"""
    
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # 获取行和列描述的 embedding
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    # 初始化集合来存储最终的行列索引
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # 针对每个阶段，获取前k行和前k列索引并添加到集合中
    for stage in solution_plan:
        # if stage['Action'].lower() == 'retrieval':
        topk = stage['Top k']
        top_rows, top_cols = retrieve_top_relevant_rows_cols(
            stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk
        )
        # top_rows = [14,15,16]
        # top_cols = [0,1,2,3,4]

        # 调用字符匹配召回函数
        # matching_rows_sub_question = retrieve_rows_by_string_match(indexed_table, stage['Sub-Level-Question'])
        # matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)

        # 更新 matching_rows
        # match_row_indices.extend(matching_rows_sub_question)
        # match_row_indices.extend(matching_rows_question)

        # 更新 embedding_rows
        embedding_row_indices.extend(top_rows)
        embedding_col_indices.extend(top_cols)

    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    combined_rows =  embedding_row_indices + match_row_indices
    # 使用集合来去重，并转回列表以保持最终顺序
    combined_rows = list(dict.fromkeys(combined_rows))

    # # 将 每一行 的前一行和后一行也加入集合
    # for row_index in combined_rows:
    #     if row_index > 0:  # 前一行存在
    #         final_row_indices.append(row_index - 1)

    #     final_row_indices.append(row_index)

    #     if row_index < len(indexed_table) - 2:  # 后一行存在，减2是因为索引偏移
    #         final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices    # 把header这一行加进去
    final_col_indices = [-1] + final_col_indices    # 把row index这一列也加进去

    final_row_indices = list(dict.fromkeys(final_row_indices))
    final_col_indices = list(dict.fromkeys(final_col_indices))
    
    # 将集合排序
    # final_row_indices = sorted(final_row_indices)
    # final_col_indices = sorted(final_col_indices)

    # 根据合并的行列索引生成最终的子表
    final_subtable = []
    for i in final_row_indices:
        subtable_row = [indexed_table[i+1][j+1] for j in final_col_indices]   # i+1 因为之前存储的行的索引没有算上第一行列名, j+1因为增加了row index这一列
        final_subtable.append(subtable_row)

    return final_subtable, final_row_indices, final_col_indices


def retrieve_final_subtable_add(solution_plan, indexed_table, row_descriptions, col_descriptions, request_gpt_embedding, question):
    
    """获取所有 retrieval 阶段的前k行列索引，加入前一行和后一行，生成最终子表。"""
    
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # 获取行和列描述的 embedding
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    # 初始化集合来存储最终的行列索引
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # 基于字符匹配
    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    # 针对每个阶段，获取前k行和前k列索引并添加到集合中
    for stage in solution_plan:
        # if stage['Action'].lower() == 'retrieval':
        topk = stage['Top k']
        top_rows, top_cols = retrieve_top_relevant_rows_cols(
            stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk
        )
        # top_rows = []
        # top_cols = []

        # 调用字符匹配召回函数
        # matching_rows_sub_question = retrieve_rows_by_string_match(indexed_table, stage['Sub-Level-Question'])
        # matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)

        # 更新 matching_rows
        # match_row_indices.extend(matching_rows_sub_question)
        # match_row_indices.extend(matching_rows_question)

        # 更新 embedding_rows
        embedding_row_indices.extend(top_rows)
        embedding_col_indices.extend(top_cols)

    embedding_row_indices = sorted(embedding_row_indices, key=int)
    
    combined_rows =  embedding_row_indices + match_row_indices
    # 使用集合来去重，并转回列表以保持最终顺序
    combined_rows = list(dict.fromkeys(combined_rows))

    # 将 每一行 的前一行和后一行也加入集合
    for row_index in combined_rows:
        if row_index > 0:  # 前一行存在
            final_row_indices.append(row_index - 1)

        final_row_indices.append(row_index)

        if row_index < len(indexed_table) - 2:  # 后一行存在，减2是因为index_table是带header的，并且索引偏移
            final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices    # 把header这一行加进去
    final_col_indices = [-1, 0] + final_col_indices    # 把row index和第一列也加进去

    final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]  # 逆序去重再逆序，保留更靠后出现的index
    final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]
    
    # 将集合排序
    # final_row_indices = sorted(final_row_indices)
    # final_col_indices = sorted(final_col_indices)

    # 根据合并的行列索引生成最终的子表
    final_subtable = []
    for i in final_row_indices:
        # print(i)
        subtable_row = [indexed_table[i+1][j+1] for j in final_col_indices]   # i+1 因为之前存储的行的索引没有算上第一行列名, j+1因为增加了row index这一列
        final_subtable.append(subtable_row)
    
    return final_subtable, final_row_indices, final_col_indices


def retrieve_final_subtable_DAG(dag_plan, indexed_table, row_descriptions, col_descriptions, request_gpt_embedding, question):
    
    """获取所有 retrieval 阶段的前k行列索引，加入前一行和后一行，生成最终子表。"""
    
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # 获取行和列描述的 embedding
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    # 初始化集合来存储最终的行列索引
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # 基于字符匹配
    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    # 针对每个阶段，获取前k行和前k列索引并添加到集合中
    for stage in dag_plan:
        # if stage['Action'].lower() == 'retrieval':
        topk = stage['Top k']
        top_rows, top_cols = retrieve_top_relevant_rows_cols(
            stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk
        )
        # top_rows = []
        # top_cols = []

        # 调用字符匹配召回函数
        # matching_rows_sub_question = retrieve_rows_by_string_match(indexed_table, stage['Sub-Level-Question'])
        # matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)

        # 更新 matching_rows
        # match_row_indices.extend(matching_rows_sub_question)
        # match_row_indices.extend(matching_rows_question)

        # 更新 embedding_rows
        embedding_row_indices.extend(top_rows)
        embedding_col_indices.extend(top_cols)

    embedding_row_indices = sorted(embedding_row_indices, key=int)
    
    combined_rows =  embedding_row_indices + match_row_indices
    # 使用集合来去重，并转回列表以保持最终顺序
    combined_rows = list(dict.fromkeys(combined_rows))

    # 将 每一行 的前一行和后一行也加入集合
    for row_index in combined_rows:
        if row_index > 0:  # 前一行存在
            final_row_indices.append(row_index - 1)

        final_row_indices.append(row_index)

        if row_index < len(indexed_table) - 2:  # 后一行存在，减2是因为index_table是带header的，并且索引偏移
            final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices    # 把header这一行加进去
    final_col_indices = [-1, 0] + final_col_indices    # 把row index和第一列也加进去

    final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]  # 逆序去重再逆序，保留更靠后出现的index
    final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]
    
    # 将集合排序
    # final_row_indices = sorted(final_row_indices)
    # final_col_indices = sorted(final_col_indices)

    # 根据合并的行列索引生成最终的子表
    final_subtable = []
    for i in final_row_indices:
        # print(i)
        subtable_row = [indexed_table[i+1][j+1] for j in final_col_indices]   # i+1 因为之前存储的行的索引没有算上第一行列名, j+1因为增加了row index这一列
        final_subtable.append(subtable_row)
    
    return final_subtable, final_row_indices, final_col_indices


def retrieve_final_subtable_DAG_save_embedding(dag_plan, indexed_table, table_embeddings , question):
    
    """获取所有 retrieval 阶段的前k行列索引，加入前一行和后一行，生成最终子表。"""
    
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # 获取行和列描述的 embedding
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    # row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    # col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)
    row_embeddings = table_embeddings["row_embeddings"]
    col_embeddings = table_embeddings["col_embeddings"]

    # 初始化集合来存储最终的行列索引
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # 基于字符匹配
    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    # 针对每个阶段，获取前k行和前k列索引并添加到集合中
    for stage in dag_plan:
        # if stage['Action'].lower() == 'retrieval':
        topk = stage['Top k']
        top_rows, top_cols = retrieve_top_relevant_rows_cols(
            stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk
        )
        # top_rows = []
        # top_cols = []

        # 调用字符匹配召回函数
        # matching_rows_sub_question = retrieve_rows_by_string_match(indexed_table, stage['Sub-Level-Question'])
        # matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)

        # 更新 matching_rows
        # match_row_indices.extend(matching_rows_sub_question)
        # match_row_indices.extend(matching_rows_question)

        # 更新 embedding_rows
        embedding_row_indices.extend(top_rows)
        embedding_col_indices.extend(top_cols)

    embedding_row_indices = sorted(embedding_row_indices, key=int)
    
    combined_rows =  match_row_indices + embedding_row_indices
    # 使用集合来去重，并转回列表以保持最终顺序
    combined_rows = list(dict.fromkeys(combined_rows))

    # 将 每一行 的前一行和后一行也加入集合
    for row_index in combined_rows:
        if row_index > 0:  # 前一行存在
            final_row_indices.append(row_index - 1)

        final_row_indices.append(row_index)

        if row_index < len(indexed_table) - 2:  # 后一行存在，减2是因为index_table是带header的，并且索引偏移
            final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices    # 把header这一行加进去
    final_col_indices = [-1, 0] + final_col_indices    # 把row index和第一列也加进去

    final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]  # 逆序去重再逆序，保留更靠后出现的index
    final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]
    
    # 将集合排序
    # final_row_indices = sorted(final_row_indices)
    # final_col_indices = sorted(final_col_indices)

    # 根据合并的行列索引生成最终的子表
    final_subtable = []
    for i in final_row_indices:
        # print(i)
        subtable_row = [indexed_table[i+1][j+1] for j in final_col_indices]   # i+1 因为之前存储的行的索引没有算上第一行列名, j+1因为增加了row index这一列
        final_subtable.append(subtable_row)
    
    return final_subtable, final_row_indices, final_col_indices


def retrieve_final_subtable_add_noplan(indexed_table, row_descriptions, col_descriptions, request_gpt_embedding, question):
    
    """获取前k行列索引，加入前一行和后一行，生成最终子表。"""
    
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # 获取行和列描述的 embedding
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    # 初始化集合来存储最终的行列索引
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # 基于字符匹配
    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    # 针对每个阶段，获取前k行和前k列索引并添加到集合中
    # for stage in solution_plan:
    #     # if stage['Action'].lower() == 'retrieval':
    #     topk = stage['Top k']
    top_rows, top_cols = retrieve_top_relevant_rows_cols_notopk(
        question, row_embeddings, col_embeddings, request_gpt_embedding, header
    )
        # top_rows = []
        # top_cols = []

        # 调用字符匹配召回函数
        # matching_rows_sub_question = retrieve_rows_by_string_match(indexed_table, stage['Sub-Level-Question'])
        # matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)

        # 更新 matching_rows
        # match_row_indices.extend(matching_rows_sub_question)
        # match_row_indices.extend(matching_rows_question)

        # 更新 embedding_rows
    embedding_row_indices.extend(top_rows)
    embedding_col_indices.extend(top_cols)

    embedding_row_indices = sorted(embedding_row_indices, key=int)
    
    combined_rows =  embedding_row_indices + match_row_indices
    # 使用集合来去重，并转回列表以保持最终顺序
    combined_rows = list(dict.fromkeys(combined_rows))

    # 将 每一行 的前一行和后一行也加入集合
    for row_index in combined_rows:
        if row_index > 0:  # 前一行存在
            final_row_indices.append(row_index - 1)

        final_row_indices.append(row_index)

        if row_index < len(indexed_table) - 2:  # 后一行存在，减2是因为index_table是带header的，并且索引偏移
            final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices    # 把header这一行加进去
    final_col_indices = [-1, 0] + final_col_indices    # 把row index和第一列也加进去

    final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]  # 逆序去重再逆序，保留更靠后出现的index
    final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]
    
    # 将集合排序
    # final_row_indices = sorted(final_row_indices)
    # final_col_indices = sorted(final_col_indices)

    # 根据合并的行列索引生成最终的子表
    final_subtable = []
    for i in final_row_indices:
        # print(i)
        subtable_row = [indexed_table[i+1][j+1] for j in final_col_indices]   # i+1 因为之前存储的行的索引没有算上第一行列名, j+1因为增加了row index这一列
        final_subtable.append(subtable_row)
    
    return final_subtable, final_row_indices, final_col_indices


# if __name__ == "__main__":
#     with open("dataset/4096_test.jsonl", 'r') as f:
#         data = f.readlines()
#     error_count = 0
#     for d in data:
#         item = json.loads(d)
#         table = item["table_text"]
#         question = item["statement"]

#         with open("prompt/get_row_template.md", "r") as f:
#             row_prompt = f.read()

#         with open("prompt/get_col_template.md", "r") as f:
#             col_prompt = f.read()

#         with open("prompt/get_solution_plan.md", "r") as f:
#             plan_prompt = f.read()
        
#         try:

#             # 生成行、列的自然语言描述
#             row_descriptions = get_row_description(table, row_prompt)
#             col_descriptions = get_col_description(table, col_prompt)

#             # 生成solution plan
#             solution_plan = get_solution_plan(table, question, plan_prompt)
#             # print(solution_plan)

#             # # 检索相关子表格
#             # final_subtable, final_row_indices, final_col_indices = retrieve_final_subtable(
#             #     solution_plan, table, row_descriptions, col_descriptions, request_gpt_embedding
#             # )
#             # print(final_subtable)
#         except ValueError as e:
#             print(f"Error encountered: {e}. Skipping this iteration.")
#             error_count += 1
#             continue  # 跳过当前循环，继续下一个表格和问题的处理

#     print(error_count)

    





