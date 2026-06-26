import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", r"Qwen/Qwen2.5-0.5B")
LORA_PATH = os.getenv("LORA_PATH", r"experiments/run_1")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "60"))
DO_SAMPLE = os.getenv("DO_SAMPLE", "false").lower() == "true"
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "training").lower()
USE_EXAMPLES = os.getenv("USE_EXAMPLES", "true").lower() == "true"
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a careful financial analyst. Answer clearly and concisely. "
    "Do not provide personalized investment advice. "
    "Do not claim live market data. "
    "When uncertain, say what should be checked.",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAS_CUDA = torch.cuda.is_available()
DTYPE = torch.bfloat16 if HAS_CUDA and torch.cuda.is_bf16_supported() else torch.float16


def find_adapter_dir(path):
    adapter_config = "adapter_config.json"

    if (path / adapter_config).is_file():
        return path

    checkpoint_dirs = sorted(
        path.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1,
        reverse=True,
    )
    for checkpoint_dir in checkpoint_dirs:
        if (checkpoint_dir / adapter_config).is_file():
            return checkpoint_dir

    nested_adapters = sorted(
        path.rglob(adapter_config),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if nested_adapters:
        return nested_adapters[0].parent

    return None


def resolve_adapter_path(path):
    path = Path(path).expanduser()
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, PROJECT_ROOT / path])

    for kaggle_root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        candidates.append(kaggle_root / path)

    seen = set()
    searched = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        searched.append(str(candidate))
        adapter_dir = find_adapter_dir(candidate)
        if adapter_dir is not None:
            print(f"Using LoRA adapter from: {adapter_dir}")
            return adapter_dir

    for search_root in [PROJECT_ROOT, Path.cwd(), Path("/kaggle/working"), Path("/kaggle/input")]:
        if not search_root.exists():
            continue
        searched.append(str(search_root))
        matches = sorted(
            search_root.rglob("adapter_config.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            adapter_dir = matches[0].parent
            print(f"Using LoRA adapter found at: {adapter_dir}")
            return adapter_dir

    raise FileNotFoundError(
        "Could not find adapter_config.json. "
        "Run training first with `python main.py`, or set LORA_PATH to the folder that "
        "contains your saved LoRA adapter. That folder must contain adapter_config.json. "
        "If you loaded a Kaggle Dataset containing the trained adapter, set LORA_PATH to "
        "its path under /kaggle/input.\n"
        f"Searched: {searched}"
    )


def build_prompt(user_prompt):
    if PROMPT_STYLE == "training":
        examples = ""
        if USE_EXAMPLES:
            examples = (
                "Instruction: You are a careful financial analyst. Answer clearly and concisely.\n"
                "Input: What does a high PE ratio mean?\n"
                "Response: A high P/E ratio means investors are paying a high price for each dollar of earnings. "
                "It can suggest strong growth expectations, expensive valuation, or both. It does not automatically "
                "mean the company is overvalued; compare it with peers, growth, margins, debt, and cash flow.\n\n"
                "Instruction: You are a careful financial analyst. Answer clearly and concisely.\n"
                "Input: Are high PE companies overvalued?\n"
                "Response: Not always. A high P/E is a warning signal to investigate, not a final conclusion. "
                "The company may deserve a premium if earnings are growing quickly and the business quality is strong.\n\n"
            )

        return (
            examples +
            f"Instruction: {SYSTEM_PROMPT}\n"
            f"Input: {user_prompt}\n"
            "Response:"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"System: {SYSTEM_PROMPT}\n"
        f"User: {user_prompt}\n"
        "Assistant:"
    )


def direct_response(user_prompt):
    text = user_prompt.lower().strip(" .!?")

    greetings = {"hi", "hello", "hey", "salam", "assalamualaikum", "assalamu alaikum"}
    if text in greetings:
        return "Hello. Ask me a finance question and I will help."

    if "hate me" in text:
        return "No, I do not hate you. I am here to help you with finance questions."

    if "your master" in text:
        return "I am here to assist you respectfully, not to be controlled. Ask me a finance question and I will help."

    rude_words = {"shit", "fuck", "bitch", "stupid"}
    if any(word in text.split() for word in rude_words):
        return "I can help, but please keep the conversation respectful."

    return None


def clean_answer(answer):
    stop_texts = [
        "\nInstruction:",
        "\nInput:",
        "\nResponse:",
        "\nUser:",
        "\nAssistant:",
        "\nQuestion:",
        "\nAnswer:",
        "\nYou are ",
        "\nPlease answer",
    ]
    for stop_text in stop_texts:
        if stop_text in answer:
            answer = answer.split(stop_text, 1)[0]
    return answer.strip().strip('"')


def generate_answer(user_prompt, sample=False):
    model_prompt = build_prompt(user_prompt)
    inputs = tokenizer(
        model_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to(device)

    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.12,
        "num_beams": 1,
    }
    if sample:
        generation_args.update({"temperature": 0.25, "top_p": 0.9})

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **generation_args,
        )

    answer_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return clean_answer(tokenizer.decode(answer_tokens, skip_special_tokens=True))


print("Loading base model...")

print("Loading LoRA adapter...")
adapter_path = resolve_adapter_path(LORA_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=DTYPE if HAS_CUDA else torch.float32,
    device_map="auto" if HAS_CUDA else "cpu",
)
base_model.config.use_cache = True

tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").is_file() else BASE_MODEL
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, adapter_path)

model.eval()
device = next(model.parameters()).device

print("Ready!\n")
print("Type your message and press Enter. Type 'exit' to quit.\n")
print(
    f"Device: {device}. Max new tokens: {MAX_NEW_TOKENS}. "
    f"Sampling: {DO_SAMPLE}. Prompt style: {PROMPT_STYLE}. Examples: {USE_EXAMPLES}.\n"
)

while True:
    prompt = input("You: ").strip()

    if prompt.lower() in {"exit", "quit"}:
        break

    if not prompt:
        continue

    answer = direct_response(prompt)
    if answer is None:
        answer = generate_answer(prompt, sample=DO_SAMPLE)

    if not answer:
        fallback_prompt = f"Give a short, direct finance analyst response to this question: {prompt}"
        answer = generate_answer(fallback_prompt, sample=True)

    if not answer:
        answer = "I am not sure how to answer that. Please ask a clear finance question."

    print(f"Assistant: {answer}\n")
