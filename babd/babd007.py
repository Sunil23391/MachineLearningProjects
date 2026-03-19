import torch
import torch.nn as nn
import torch.optim as optim
from text_processor import TextProcessor

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Embedding(input_dim, output_dim)
        self.transformer = nn.Transformer(output_dim, nhead, num_layers)
        self.decoder = nn.Linear(output_dim, input_dim)
        self.output_dim = output_dim

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_mask=None):
        src = self.encoder(src) * (self.output_dim ** 0.5)
        tgt = self.encoder(tgt) * (self.output_dim ** 0.5)
        output = self.transformer(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_mask=memory_mask)
        output = self.decoder(output)
        return output

def process_text(text_processor, sentence):
    array, attention_mask_array, position_array = text_processor.process_sentence(sentence)
    tensor = torch.tensor(array)
    attention_mask = torch.tensor(attention_mask_array, dtype=torch.float)
    return tensor, attention_mask

# Read text from file
with open('input_text_001.txt', 'r') as file:
    text = file.read()

# Create an instance of TextProcessor
processor = TextProcessor(text)

# Tokenize the text into sentences
sentences = processor.tokenize(text)


# Create transformer model
input_dim = len(processor.vocab)
output_dim = 512
nhead = 8
num_layers = 6
model = TransformerModel(input_dim, output_dim, nhead, num_layers)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = 100
current_sentence = 1
while len(sentences) - 1 > current_sentence:
    # Process the first and second sentences using the common function
    input_tensor, input_attention_mask = process_text(processor, sentences[current_sentence-1])
    output_tensor, output_attention_mask = process_text(processor, sentences[current_sentence])
    # Train the model (example for 10 epochs)
    epoch = 0
    print('epoch starts',epoch)
    while loss >= 0.4 and epoch <= 20:
        epoch += 1
        model.train()
        optimizer.zero_grad()
        output = model(input_tensor, output_tensor, input_attention_mask, output_attention_mask)
        loss = criterion(output.view(-1, input_dim), output_tensor.view(-1))
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}')

# Output the processed input and output arrays
# input_string = processor.decode(input_array)
# output_string = processor.decode(output_array)

# print("Input array:", input_array)
# print("Output array:", output_array)
# print("Input string:", input_string)
# print("Output string:", output_string)
