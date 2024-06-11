from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


# Prepare input messages for the chat
messages = [
    {"role": "system", "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}
]

# Tokenize the input messages and set return_tensors to "pt" (PyTorch tensors)
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")


# Generate responses using the model
outputs = model.generate(inputs, max_new_tokens=32)

# Decode and print the generated text
text = tokenizer.batch_decode(outputs)[0]
print(text)
