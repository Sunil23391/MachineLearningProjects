import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize

# Download the NLTK tokenizer data
nltk.download('punkt')

# Given text
text = '1. What is ReactJS?\nReactJS is a JavaScript library used to build reusable components for the view layer in MVC architecture. It is used to build the Single Page Application (SPA) due to its component-based architecture, efficient re-rendering with the Virtual DOM, and ability to manage dynamic content without needing full page reloads. It is written in JSX.'

# Extract the first sentence (input) and second sentence (output)
sentences = text.split('\n')
input_text = sentences[0]
output_text = sentences[1]

# Tokenize the whole paragraph into words
tokens = word_tokenize(text)
vocab = sorted(set(tokens))
token_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_token = {idx: word for word, idx in token_to_index.items()}

# Tokenize the text
tokenized_text = [token_to_index[word] for word in tokens]

# Parameters
sequence_length = len(tokenized_text)
vocab_size = len(vocab)
embedding_dim = 48  # Embedding size (C)
num_heads = 3  # Number of attention heads
head_dim = embedding_dim // num_heads  # Dimension per head
ffn_dim = 192  # Feed-Forward Neural Network dimension


# Token embedding matrix
token_embedding = nn.Embedding(vocab_size, embedding_dim)

# Position embedding matrix
position_embedding = nn.Embedding(sequence_length, embedding_dim)

# Example usage
# Tokenize the input text
tokenized_input_text = [token_to_index[word] for word in word_tokenize(input_text)]

# Convert to tensor
input_tensor = torch.tensor(tokenized_input_text)

# Embed the tokens
input_embeddings = token_embedding(input_tensor)

# Generate position embeddings
position_indices = torch.arange(len(tokenized_input_text)).unsqueeze(0)
position_embeddings = position_embedding(position_indices)

# Add position embeddings to token embeddings
input_embeddings_with_position = input_embeddings + position_embeddings

print(input_embeddings_with_position)