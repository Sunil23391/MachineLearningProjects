import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download the NLTK tokenizer data
nltk.download('punkt')

# Given text
text = 'What is ReactJS?\nReactJS is a JavaScript library used to build reusable components for the view layer in MVC architecture. It is used to build the Single Page Application (SPA) due to its component-based architecture, efficient re-rendering with the Virtual DOM, and ability to manage dynamic content without needing full page reloads. It is written in JSX.'

# Parameters
sequence_length = 100
embedding_dim = 48  # Embedding size (C)
num_heads = 3  # Number of attention heads
head_dim = embedding_dim // num_heads  # Dimension per head
ffn_dim = 192  # Feed-Forward Neural Network dimension

# Split the given text into sentences
sentences = sent_tokenize(text)

# Tokenize the whole paragraph into words and build vocabulary
tokens = word_tokenize(text)
vocab = sorted(set(tokens))

# Add padding token to the vocabulary
padding_token = '<PAD>'
vocab.insert(0, padding_token)
token_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_token = {idx: word for word, idx in token_to_index.items()}

# Generate position indices
position_indices = list(range(sequence_length))

def process_sentence(sentence):
    # Tokenize the input sentence
    tokenized_input_text = [token_to_index[word] for word in word_tokenize(sentence)]

    # Pad the sequence to the specified length
    padded_input_text = tokenized_input_text + [token_to_index[padding_token]] * (sequence_length - len(tokenized_input_text))
    input_array = padded_input_text[:sequence_length]

    # Create the attention mask
    attention_mask_array = [1 if i < len(tokenized_input_text) else 0 for i in range(sequence_length)]

    # Create position array (same as position indices)
    position_array = position_indices

    return input_array, attention_mask_array, position_array

# Example usage
input_text = sentences[0]
output_text = sentences[1]
input_array, attention_mask_array, position_array = process_sentence(input_text)
output_array, _, _ = process_sentence(output_text)

# print("Input array:", input_array)
# print("Attention mask array:", attention_mask_array)
# print("Position array:", position_array)

def decode(output_array):
    decoded_tokens = [index_to_token[idx] for idx in output_array if idx in index_to_token and index_to_token[idx] != padding_token]
    decoded_string = ' '.join(decoded_tokens)
    return decoded_string

input_string = decode(input_array)
output_string = decode(output_array)

# Convert input_array indices back to tokens for verification
print("Input string:", input_string)
print("Output string:", output_string)
