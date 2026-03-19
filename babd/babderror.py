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

# Define the Transformer components
W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_O = nn.Linear(embedding_dim, embedding_dim, bias=False)  # Output projection

# Define the MLP linear layers
mlp_linear1 = nn.Linear(embedding_dim, ffn_dim)
mlp_linear2 = nn.Linear(ffn_dim, embedding_dim)

# Define Layer Normalization layers
layer_norm1 = nn.LayerNorm(embedding_dim)
layer_norm2 = nn.LayerNorm(embedding_dim)

# Define the LM head
lm_head = nn.Linear(embedding_dim, vocab_size)

# Transformer block
def transformer_block(input_embeddings):
    # Apply Layer Normalization to the input embeddings
    normalized_input_embeddings = layer_norm1(input_embeddings)

    # Compute Query, Key, and Value matrices
    Q = W_Q(normalized_input_embeddings)  # Shape (T, C)
    K = W_K(normalized_input_embeddings)  # Shape (T, C)
    V = W_V(normalized_input_embeddings)  # Shape (T, C)

    # Split Q, K, V into multiple heads
    Q = Q.view(sequence_length, num_heads, head_dim).transpose(0, 1)  # Shape (num_heads, T, head_dim)
    K = K.view(sequence_length, num_heads, head_dim).transpose(0, 1)  # Shape (num_heads, T, head_dim)
    V = V.view(sequence_length, num_heads, head_dim).transpose(0, 1)  # Shape (num_heads, T, head_dim)

    # Compute attention scores for each head
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))  # Shape (num_heads, T, T)

    # Apply attention mask (all ones for simplicity)
    attention_mask = torch.ones(num_heads, sequence_length, sequence_length)
    attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

    # Compute attention weights for each head
    attention_weights = F.softmax(attention_scores, dim=-1)  # Shape (num_heads, T, T)

    # Compute attention output for each head
    attention_output = torch.matmul(attention_weights, V)  # Shape (num_heads, T, head_dim)

    # Concatenate the outputs of all heads
    concatenated_attention_output = attention_output.transpose(0, 1).contiguous().view(sequence_length, embedding_dim)  # Shape (T, C)

    # Apply the projection linear transformation
    final_output = W_O(concatenated_attention_output)  # Shape (T, C)

    # Add the attention output to the input (residual connection)
    attention_residual = final_output + input_embeddings

    # Apply Layer Normalization to the attention residual
    normalized_attention_residual = layer_norm2(attention_residual)

    # Apply the MLP linear transformations
    mlp_output = mlp_linear1(normalized_attention_residual)
    mlp_activation = F.gelu(mlp_output)
    mlp_projection_output = mlp_linear2(mlp_activation)

    # Add the MLP output to the attention residual (residual connection)
    mlp_residual = mlp_projection_output + normalized_attention_residual

    return mlp_residual

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

# Combine token and position embeddings
combined_embeddings = input_embeddings + position_embeddings.squeeze(0)

# Pass through the transformer block
output_embeddings = transformer_block(combined_embeddings)

# Apply the LM head to get logits
logits = lm_head(output_embeddings)

# Calculate softmax of the logits
logits_softmax = F.softmax(logits, dim=-1)

# Output the logits softmax
print(logits_softmax)
