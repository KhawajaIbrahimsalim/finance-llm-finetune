def generate(model, tokenizer, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)