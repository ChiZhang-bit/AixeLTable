import os
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from flask import Flask, request, jsonify

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# define logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


port_map = {
    'llama': 6666,
    'mistral': 6667,
    'phi': 6668,
    'QWen-7B': 6671
}


# call api test
# curl -X POST http://localhost:6667/generate -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you?"}'
# curl -X POST http://localhost:6667/generate -H "Content-Type: application/json" -d '{"prompt": "Who are you?"}'

# define server app
app = Flask(__name__)

model_path = ''


port = port_map['mistral'] #pid: 3495351

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2'
)
pad_token_id = tokenizer.eos_token_id  # 设置结束符
# generate api
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 8192)
    temperature = data.get('temperature', 0)
    do_sample = data.get('do_sample', False)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 1.0) 

    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    logger.info(f'Input prompt: {formatted_prompt}')

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        pad_token_id = tokenizer.eos_token_id
    )

    generated_tokens = outputs[0, inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logger.info(f'Generated response: {response}')

    return jsonify({'response': response})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
