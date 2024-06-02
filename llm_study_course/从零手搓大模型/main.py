from transformers.models.qwen2 import Qwen2Config,Qwen2Model
import torch


def run_llama():
    llamaConfig = Qwen2Config(
        vocab_size= 151936,
        hidden_size = 4096//2,
        intermediate_size=22016//2,
        num_hidden_layers=32//2,
        num_attention_heads=32//2,
        max_position_embeddings=2048//2
    )
    llamaModel = Qwen2Model(config=llamaConfig)
    input_ids = torch.randint(0,llamaConfig.vocab_size,(4,30))

    res = llamaModel(input_ids)
    print(res)

if __name__ == '__main__':
    run_llama()