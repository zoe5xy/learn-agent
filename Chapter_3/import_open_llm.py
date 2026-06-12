import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify model ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# Set device, prioritize GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model and move it to specified device
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("Model and tokenizer loaded!")

# Prepare dialogue input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, please introduce yourself,what's your name?"}
]

# Use tokenizer's template to format input
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Encode input text
model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("Encoded input text:")
print(model_inputs)

# Use model to generate answer
# max_new_tokens controls the maximum number of new Tokens the model can generate
generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,  # 新增这一行
    max_new_tokens=512
)
# Truncate the input part from generated Token IDs
# This way we only decode the newly generated part by the model
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode generated Token IDs
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\nModel's answer:")
print(response)

