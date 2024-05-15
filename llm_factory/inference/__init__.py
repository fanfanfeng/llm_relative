from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def single_local_test(model_path):

    model = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto',load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("prompt text:\n",text)

    model_inputs = tokenizer([text],return_tensors='pt').to(try_gpu())

    outputs = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    response = outputs[0][model_inputs.input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))



if __name__ == '__main__':
    model_path = r'G:\llm\models\Qwen1.5-7B-Chat\Qwen1.5-7B-Chat'
    single_local_test(model_path)