# guarantee json output
import os
import logging
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from flask import Flask, request, jsonify

from pydantic import BaseModel
import outlines




# define logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# call api test
# curl -X POST http://localhost:6667/generate -H "Content-Type: application/json" -d '{"prompt": "Generate data to describe a person, including name, age, and gender.", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}, "gender": {"type": "string"}}}}'

# define server app
app = Flask(__name__)



os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# model_path = 'hf_models/Meta-Llama-3.1-8B-Instruct'
# port = 6666

model_path = 'hf_models/Mistral-Nemo-Instruct-2407'
port = 6667

# model_path = 'hf_models/Phi-3.5-mini-instruct'
# port = 6668

print(f'*** Loading model - {model_path}...')
print(f'*** Launching server on port {port}...')


# load model, tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2'
)



model = outlines.models.Transformers(model, tokenizer)



# generate api
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    schema = data.get('schema', {})
    max_tokens = data.get('max_tokens', 150)
    temperature = data.get('temperature', 0)
    do_sample = data.get('do_sample', False)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 1.0)
    
    if temperature == 0:
        temperature = 1e-4

    if not isinstance(schema, dict):
        return jsonify({'error': 'Invalid schema provided'}), 400

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    logger.info(f'Input prompt: {prompt}')
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    schema_str = json.dumps(schema)

    sampler = outlines.samplers.multinomial(
        temperature=temperature,
        # top_k=top_k,
        # top_p=top_p
    )
    generator = outlines.generate.json(model, schema_str, sampler=sampler)
    response = generator(formatted_prompt)
    response = json.dumps(response)

    logger.info(f'Generated response: {response}')

    return jsonify({'response': response})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)