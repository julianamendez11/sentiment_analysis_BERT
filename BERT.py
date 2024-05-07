from transformers import BertModel, BertTokenizer
import torch

# Load Pre-trained Model
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Data Preprocessing
tokenized_texts = [tokenizer.encode(review, add_special_tokens=True, max_length=512) for review in reviews]
max_len = max(len(review) for review in tokenized_texts)
padded_tokenized_texts = [review + [0] * (max_len - len(review)) for review in tokenized_texts]

# Convert to Tensors
input_ids = torch.tensor(padded_tokenized_texts)

# Get Intermediate Representations
with torch.no_grad():
    outputs = model(input_ids)
    hidden_states = outputs[2]  # Hidden representations are at position 2

# Get Insights
# You can use the hidden representations to analyze similarities between reviews,
# identify common themes, or any other type of analysis you want to perform.
