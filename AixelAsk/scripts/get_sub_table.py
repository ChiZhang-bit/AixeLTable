import json
import sys
import re
from utils.request_gpt import request_gpt_chat, request_gpt_embedding
from scripts.processing_format import get_row_description, get_col_description
from scripts.generate_solution_plan import get_solution_plan
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from scripts.schema_linking import rewrite_question


def get_embeddings(descriptions, request_gpt_embedding):
    """Get the embedding for each description."""
    embeddings = [request_gpt_embedding(desc) for desc in tqdm(descriptions, desc="Generating Embeddings")]
    return embeddings


# def map_descriptions_to_table_indices(row_descriptions, col_descriptions):
#     """Build a mapping from row/column descriptions to original table indices."""
#     row_mapping = {desc: i for i, desc in enumerate(row_descriptions)}
#     col_mapping = {desc: j for j, desc in enumerate(col_descriptions)}
#     return row_mapping, col_mapping


def retrieve_rows_by_string_match(table, question):
    """
    Find row indices related to the question via string matching.
    """
    # Remove trailing punctuation and lowercase the question
    question_cleaned = re.sub(r'[^\w\s]$', '', question.strip()).lower()
    question_words = set(word for word in re.split(r'\W+', question_cleaned) if word)

    matching_rows = set()

    for row_index, row in enumerate(reversed(table[1:])):
        actual_row_index = len(table) - row_index - 2  # -2 because we exclude the header row
        for cell in row:
            # Lowercase cell content and split by non-alphanumeric characters
            cell_words = set(
                word for word in re.split(r'\W+', str(cell).lower())
                if (len(word) > 3 or word.isdigit()) and word
            )

            # Check if there is an intersection between cell_words and question_words
            if question_words.intersection(cell_words):
                # print(f"Matched row index: {actual_row_index}, cell content: '{cell}', in question: '{question}'")
                matching_rows.add(actual_row_index)
    return list(matching_rows)


def retrieve_top_relevant_rows_cols(stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk):
    """For each retrieval stage, rank all rows/columns by similarity and select the top-k rows and top-5 columns."""
    if topk == 'all':
        topk = len(row_embeddings)
    else:
        topk = int(topk)

    sub_level_question = stage['Sub-Level-Question']
    rewrited_sub_level_question = rewrite_question(sub_level_question, header)
    question_embedding = request_gpt_embedding(rewrited_sub_level_question)

    # Compute similarities with row descriptions and sort
    row_similarities = cosine_similarity([question_embedding], row_embeddings)[0]
    sorted_row_indices = np.argsort(-row_similarities)  # Descending

    num_rows_to_select = min(topk, len(sorted_row_indices))
    top_sorted_rows = sorted_row_indices[:num_rows_to_select]

    # Compute similarities with column descriptions and sort
    col_similarities = cosine_similarity([question_embedding], col_embeddings)[0]
    sorted_col_indices = np.argsort(-col_similarities)  # Descending

    num_cols_to_select = min(5, len(sorted_col_indices))
    top_sorted_cols = sorted_col_indices[:num_cols_to_select]

    return top_sorted_rows, top_sorted_cols


def retrieve_top_relevant_rows_cols_notopk(question, row_embeddings, col_embeddings, request_gpt_embedding, header):
    """Rank all rows/columns by similarity and select top 60 rows and top 5 columns (no explicit top-k input)."""
    rewrited_sub_level_question = rewrite_question(question, header)
    question_embedding = request_gpt_embedding(rewrited_sub_level_question)

    # Rows
    row_similarities = cosine_similarity([question_embedding], row_embeddings)[0]
    sorted_row_indices = np.argsort(-row_similarities)
    num_rows_to_select = min(60, len(sorted_row_indices))
    top_sorted_rows = sorted_row_indices[:num_rows_to_select]

    # Columns
    col_similarities = cosine_similarity([question_embedding], col_embeddings)[0]
    sorted_col_indices = np.argsort(-col_similarities)
    num_cols_to_select = min(5, len(sorted_col_indices))
    top_sorted_cols = sorted_col_indices[:num_cols_to_select]

    return top_sorted_rows, top_sorted_cols


def retrieve_final_subtable(solution_plan, indexed_table, row_descriptions, col_descriptions, request_gpt_embedding, question):
    """Collect top-k row/column indices across all retrieval stages and build the final subtable."""
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # Row/column description embeddings
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    # Containers for final indices
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # For each stage, collect top-k rows and top columns
    for stage in solution_plan:
        # if stage['Action'].lower() == 'retrieval':
        topk = stage['Top k']
        top_rows, top_cols = retrieve_top_relevant_rows_cols(
            stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk
        )
        # top_rows = [14,15,16]
        # top_cols = [0,1,2,3,4]

        # String-match recall (optional)
        # matching_rows_sub_question = retrieve_rows_by_string_match(indexed_table, stage['Sub-Level-Question'])
        # matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
        # match_row_indices.extend(matching_rows_sub_question)
        # match_row_indices.extend(matching_rows_question)

        # Embedding-based rows/cols
        embedding_row_indices.extend(top_rows)
        embedding_col_indices.extend(top_cols)

    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    combined_rows = embedding_row_indices + match_row_indices
    # Deduplicate while preserving order
    combined_rows = list(dict.fromkeys(combined_rows))

    # Optionally include the previous/next row for context
    # for row_index in combined_rows:
    #     if row_index > 0:
    #         final_row_indices.append(row_index - 1)
    #     final_row_indices.append(row_index)
    #     if row_index < len(indexed_table) - 2:  # minus 2 for header offset
    #         final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices  # include header row
    final_col_indices = [-1] + final_col_indices  # include row-index column

    final_row_indices = list(dict.fromkeys(final_row_indices))
    final_col_indices = list(dict.fromkeys(final_col_indices))

    # Build the final subtable from collected indices
    final_subtable = []
    for i in final_row_indices:
        # i+1 and j+1 because indexed_table includes header and row-index columns
        subtable_row = [indexed_table[i + 1][j + 1] for j in final_col_indices]
        final_subtable.append(subtable_row)

    return final_subtable, final_row_indices, final_col_indices


def retrieve_final_subtable_add(solution_plan, indexed_table, row_descriptions, col_descriptions, request_gpt_embedding, question):
    """Collect top-k row/column indices for each retrieval stage, add previous/next rows for context, and build the final subtable."""
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # Row/column description embeddings
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    # Containers for final indices
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # String-match based recall
    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    # For each stage, collect top-k rows and cols
    for stage in solution_plan:
        # if stage['Action'].lower() == 'retrieval':
        topk = stage['Top k']
        top_rows, top_cols = retrieve_top_relevant_rows_cols(
            stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk
        )
        # top_rows = []
        # top_cols = []

        # Embedding-based rows/cols
        embedding_row_indices.extend(top_rows)
        embedding_col_indices.extend(top_cols)

    embedding_row_indices = sorted(embedding_row_indices, key=int)

    combined_rows = embedding_row_indices + match_row_indices
    combined_rows = list(dict.fromkeys(combined_rows))  # dedupe preserve order

    # Add the previous and next row for each selected row to provide context
    for row_index in combined_rows:
        if row_index > 0:  # previous row exists
            final_row_indices.append(row_index - 1)

        final_row_indices.append(row_index)

        # minus 2 because indexed_table includes header and an offset in indices
        if row_index < len(indexed_table) - 2:  # next row exists
            final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices  # include header row
    final_col_indices = [-1, 0] + final_col_indices  # include row-index and first column

    # Reverse-deduplicate to keep later occurrences
    final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]
    final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]

    # Build final subtable
    final_subtable = []
    for i in final_row_indices:
        # print(i)
        subtable_row = [indexed_table[i + 1][j + 1] for j in final_col_indices]
        final_subtable.append(subtable_row)

    return final_subtable, final_row_indices, final_col_indices


def retrieve_final_subtable_DAG(dag_plan, indexed_table, row_descriptions, col_descriptions, request_gpt_embedding, question):
    """Collect top-k row/column indices for each DAG retrieval stage, add context rows, and build the final subtable."""
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # Row/column description embeddings
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    # Containers for final indices
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # String-match recall
    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    # For each DAG stage
    for stage in dag_plan:
        # if stage['Action'].lower() == 'retrieval':
        topk = stage['Top k']
        top_rows, top_cols = retrieve_top_relevant_rows_cols(
            stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk
        )
        # top_rows = []
        # top_cols = []

        # Embedding-based
        embedding_row_indices.extend(top_rows)
        embedding_col_indices.extend(top_cols)

    embedding_row_indices = sorted(embedding_row_indices, key=int)

    combined_rows = embedding_row_indices + match_row_indices
    combined_rows = list(dict.fromkeys(combined_rows))

    # Add neighboring rows for context
    for row_index in combined_rows:
        if row_index > 0:
            final_row_indices.append(row_index - 1)

        final_row_indices.append(row_index)

        if row_index < len(indexed_table) - 2:  # adjusted for header and index offset
            final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices  # include header row
    final_col_indices = [-1, 0] + final_col_indices  # include row-index and first column

    final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]
    final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]

    # Build final subtable
    final_subtable = []
    for i in final_row_indices:
        # print(i)
        subtable_row = [indexed_table[i + 1][j + 1] for j in final_col_indices]
        final_subtable.append(subtable_row)

    return final_subtable, final_row_indices, final_col_indices


def retrieve_final_subtable_DAG_save_embedding(dag_plan, indexed_table, table_embeddings, question):
    """Collect top-k row/column indices for each DAG retrieval stage (using precomputed embeddings), add context rows, and build the final subtable."""
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # Precomputed embeddings
    # row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    # col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)
    row_embeddings = table_embeddings["row_embeddings"]
    col_embeddings = table_embeddings["col_embeddings"]

    # Containers for final indices
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # String-match recall
    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    # For each DAG stage
    for stage in dag_plan:
        # if stage['Action'].lower() == 'retrieval':
        topk = stage['Top k']
        top_rows, top_cols = retrieve_top_relevant_rows_cols(
            stage, row_embeddings, col_embeddings, request_gpt_embedding, header, topk
        )
        # top_rows = []
        # top_cols = []

        # Embedding-based
        embedding_row_indices.extend(top_rows)
        embedding_col_indices.extend(top_cols)

    embedding_row_indices = sorted(embedding_row_indices, key=int)

    combined_rows = match_row_indices + embedding_row_indices
    combined_rows = list(dict.fromkeys(combined_rows))

    # Add neighboring rows for context
    for row_index in combined_rows:
        if row_index > 0:
            final_row_indices.append(row_index - 1)

        final_row_indices.append(row_index)

        if row_index < len(indexed_table) - 2:  # adjusted for header and index offset
            final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices  # include header row
    final_col_indices = [-1, 0] + final_col_indices  # include row-index and first column

    final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]
    final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]

    # Build final subtable
    final_subtable = []
    for i in final_row_indices:
        # print(i)
        subtable_row = [indexed_table[i + 1][j + 1] for j in final_col_indices]
        final_subtable.append(subtable_row)

    return final_subtable, final_row_indices, final_col_indices


def retrieve_final_subtable_add_noplan(indexed_table, row_descriptions, col_descriptions, request_gpt_embedding, question):
    """Collect top row/column indices (no explicit plan), add context rows, and build the final subtable."""
    header_with_index = indexed_table[0]
    header = header_with_index[1:]

    # Row/column description embeddings
    # row_embeddings = [[]]
    # col_embeddings = [[]]
    row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
    col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

    # Containers for final indices
    final_row_indices = list()
    final_col_indices = list()
    match_row_indices = list()
    embedding_row_indices = list()
    embedding_col_indices = list()

    # String-match recall
    matching_rows_question = retrieve_rows_by_string_match(indexed_table, question)
    match_row_indices.extend(matching_rows_question)

    # No plan: directly compute top rows/cols
    top_rows, top_cols = retrieve_top_relevant_rows_cols_notopk(
        question, row_embeddings, col_embeddings, request_gpt_embedding, header
    )

    # Embedding-based rows/cols
    embedding_row_indices.extend(top_rows)
    embedding_col_indices.extend(top_cols)

    embedding_row_indices = sorted(embedding_row_indices, key=int)

    combined_rows = embedding_row_indices + match_row_indices
    combined_rows = list(dict.fromkeys(combined_rows))

    # Add neighboring rows for context
    for row_index in combined_rows:
        if row_index > 0:
            final_row_indices.append(row_index - 1)

        final_row_indices.append(row_index)

        # adjusted for header and index offset
        if row_index < len(indexed_table) - 2:
            final_row_indices.append(row_index + 1)

    final_col_indices.extend(embedding_col_indices)
    final_row_indices = [-1] + final_row_indices  # include header row
    final_col_indices = [-1, 0] + final_col_indices  # include row-index and first column

    final_row_indices = list(dict.fromkeys(final_row_indices[::-1]))[::-1]  # reverse-dedupe to keep the later index
    final_col_indices = list(dict.fromkeys(final_col_indices[::-1]))[::-1]

    # Build final subtable
    final_subtable = []
    for i in final_row_indices:
        # print(i)
        subtable_row = [indexed_table[i + 1][j + 1] for j in final_col_indices]
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
#
#         with open("prompt/get_row_template.md", "r") as f:
#             row_prompt = f.read()
#
#         with open("prompt/get_col_template.md", "r") as f:
#             col_prompt = f.read()
#
#         with open("prompt/get_solution_plan.md", "r") as f:
#             plan_prompt = f.read()
#
#         try:
#             # Generate row/column descriptions
#             row_descriptions = get_row_description(table, row_prompt)
#             col_descriptions = get_col_description(table, col_prompt)
#
#             # Generate solution plan
#             solution_plan = get_solution_plan(table, question, plan_prompt)
#             # print(solution_plan)
#
#             # # Retrieve related subtable
#             # final_subtable, final_row_indices, final_col_indices = retrieve_final_subtable(
#             #     solution_plan, table, row_descriptions, col_descriptions, request_gpt_embedding
#             # )
#             # print(final_subtable)
#         except ValueError as e:
#             print(f"Error encountered: {e}. Skipping this iteration.")
#             error_count += 1
#             continue
#
#     print(error_count)
