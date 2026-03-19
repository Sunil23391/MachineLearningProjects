from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset
emotion_dataset = load_dataset('emotion')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Define the preprocess function
def preprocess_function(examples):
    # Ensure tokenizer is accessible
    assert tokenizer is not None, "Tokenizer is not defined"
    return tokenizer([" ".join(x) for x in examples['text']])

# Apply the preprocess function to the dataset
tokenized_emotion_dataset = emotion_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=emotion_dataset['train'].column_names,
)

print(tokenized_emotion_dataset)