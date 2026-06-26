import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def load_model(model_name):
    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda and torch.cuda.is_bf16_supported() else torch.float16

    if not has_cuda:
        print("WARNING: CUDA GPU not detected. Training will run on CPU and will be very slow.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if has_cuda else torch.float32,
        device_map="auto" if has_cuda else "cpu",
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if has_cuda:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
