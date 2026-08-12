from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("theschoolofai/BrahmicTokenizer-131K")

print(tokenizer.encode("भारत एक देश है", add_special_tokens=False))
# -> [66526, 2420, 13092, 732]

print(tokenizer.encode("1234567890", add_special_tokens=False))
# -> [4660, 14932, 23133, 26]