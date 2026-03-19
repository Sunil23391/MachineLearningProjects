import torch
import torch.nn as nn
import torch.optim as optim
from text_processor import TextProcessor
import signal
import sys

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
        logits = self.decoder(output)
        return logits

def process_text(text_processor, sentence):
    array, attention_mask_array, position_array = text_processor.process_sentence(sentence)
    tensor = torch.tensor(array)
    attention_mask = torch.tensor(attention_mask_array, dtype=torch.float)
    return tensor, attention_mask

def signal_handler(sig, frame):
    global halt_training
    halt_training = True
    print('Training halted. You can restart it by entering "resume" in the command prompt.')

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def train_model(model, criterion, optimizer, input_tensor, output_tensor, input_attention_mask, output_attention_mask):
    global halt_training
    epoch = 0
    loss = 100
    while loss >= 0.4 and epoch <= 20:
        epoch += 1
        model.train()
        optimizer.zero_grad()
        logits = model(input_tensor, output_tensor, input_attention_mask, output_attention_mask)
        softmax_output = torch.softmax(logits, dim=-1)
        final_output = torch.argmax(softmax_output, dim=-1)
        loss = criterion(logits.view(-1, input_dim), output_tensor.view(-1))
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/20], Loss: {loss.item():.4f}')
        if halt_training:
            break
    return final_output

def generate_output_from_input(model, text_processor, input_sentence):
    input_tensor, input_attention_mask = process_text(text_processor, input_sentence)
    input_tensor = input_tensor
    input_attention_mask = input_attention_mask
    output = model(input_tensor, input_tensor, input_attention_mask, input_attention_mask)
    softmax_output = torch.softmax(output, dim=-1)
    final_output = torch.argmax(softmax_output, dim=-1)
    final_output_text = text_processor.decode(final_output.squeeze().numpy())
    return final_output_text

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
current_sentence = 1

halt_training = False

while len(sentences) - 1 > current_sentence:
    # Process the first and second sentences using the common function
    input_tensor, input_attention_mask = process_text(processor, sentences[current_sentence-1])
    output_tensor, output_attention_mask = process_text(processor, sentences[current_sentence])

    print('epoch starts', current_sentence)
    final_output = train_model(model, criterion, optimizer, input_tensor, output_tensor, input_attention_mask, output_attention_mask)
    
    if halt_training:
        command = input('Enter "resume" to continue training or input text to generate output: ')
        if command == "resume":
            halt_training = False
            final_output = train_model(model, criterion, optimizer, input_tensor, output_tensor, input_attention_mask, output_attention_mask)
        else:
            user_input_text = command
            generated_output_text = generate_output_from_input(model, processor, user_input_text)
            print("Generated output text:", generated_output_text)

    current_sentence += 1

# Output the processed input and output arrays
# input_string = processor.decode(input_array)
# output_string = processor.decode(output_array)

# print("Input array:", input_array)
# print("Output array:", output_array)
# print("Input string:", input_string)
# print("Output string:", output_string)
