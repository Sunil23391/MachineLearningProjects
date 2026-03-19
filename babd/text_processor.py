import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download the NLTK tokenizer data
nltk.download('punkt')

class TextProcessor:
    def __init__(self, text, sequence_length=100):
        self.text = text
        self.sequence_length = sequence_length
        self.padding_token = '<PAD>'
        
        # Tokenize the whole paragraph into words and build vocabulary
        tokens = word_tokenize(text)
        vocab = sorted(set(tokens))
        self.vocab = vocab
        
        # Add padding token to the vocabulary
        vocab.insert(0, self.padding_token)
        self.token_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_token = {idx: word for word, idx in self.token_to_index.items()}
        
        # Generate position indices
        self.position_indices = list(range(sequence_length))
    
    def tokenize(self, text):
        """Tokenizes the given text into sentences and words."""
        sentences = sent_tokenize(text)
        return sentences

    def process_sentence(self, sentence):
        """Processes a given sentence to generate input_array, attention_mask_array, and position_array."""
        # Tokenize the input sentence
        tokenized_input_text = [self.token_to_index[word] for word in word_tokenize(sentence)]
        
        # Pad the sequence to the specified length
        padded_input_text = tokenized_input_text + [self.token_to_index[self.padding_token]] * (self.sequence_length - len(tokenized_input_text))
        input_array = padded_input_text[:self.sequence_length]
        
        # Create the attention mask
        attention_mask_array = [1 if i < len(tokenized_input_text) else 0 for i in range(self.sequence_length)]
        
        # Create position array (same as position indices)
        position_array = self.position_indices
        
        return input_array, attention_mask_array, position_array

    def decode(self, input_array):
        """Decodes the input_array back to a string, removing the padding tokens."""
        decoded_tokens = [self.index_to_token[idx] for idx in input_array if idx in self.index_to_token and self.index_to_token[idx] != self.padding_token]
        decoded_string = ' '.join(decoded_tokens)
        return decoded_string

# Example usage
if __name__ == '__main__':
    text = 'What is ReactJS?\nReactJS is a JavaScript library used to build reusable components for the view layer in MVC architecture. It is used to build the Single Page Application (SPA) due to its component-based architecture, efficient re-rendering with the Virtual DOM, and ability to manage dynamic content without needing full page reloads. It is written in JSX.'
    
    processor = TextProcessor(text)
    sentences = processor.tokenize(text)
    
    input_text = sentences[0]
    output_text = sentences[1]
    
    input_array, attention_mask_array, position_array = processor.process_sentence(input_text)
    output_array, _, _ = processor.process_sentence(output_text)
    
    input_string = processor.decode(input_array)
    output_string = processor.decode(output_array)
    
    print("Input array:", input_array)
    print("Input string:", input_string)
    print("Output array:", output_array)
    print("Output string:", output_string)
