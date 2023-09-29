from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer from the Hugging Face model hub
model_name = "gpt2"  # You can replace this with other model names
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a prompt text
prompt = "I enjoy walking with my cute dog in the park."

# Tokenize the prompt text and obtain the output tensor
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate a sequence of tokens (words) using the model
outputs = model.generate(inputs, max_length=50, num_return_sequences=1, temperature=1.0)

# Decode the generated tokens back into a string
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)


# save model
model.save_pretrained("gpt2-model")
tokenizer.save_pretrained("gpt2-model")
# 
# # load model