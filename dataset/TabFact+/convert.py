import json

input_path = 'dataset/tabfact/large_tabfact_test_data.jsonl'
output_path = 'dataset/tabfact/large_tabfact_test_data_str.jsonl'  # 可设为 input_path 覆盖原文件

def convert_table_cells_to_str(table):
    return [[str(cell) for cell in row] for row in table]

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        item = json.loads(line)
        if 'table_text' in item:
            item['table_text'] = convert_table_cells_to_str(item['table_text'])
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write('\n')
