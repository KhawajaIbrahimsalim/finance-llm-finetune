def format_finance(example):
    return {
        "text": f"""
Instruction: You are a financial analyst. Analyze and explain clearly.
Input: {example.get('question', example.get('input', ''))}
Response: {example.get('answer', example.get('output', ''))}
"""
    }