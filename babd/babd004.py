# Import the TextProcessor class from the text_processor module
from text_processor import TextProcessor

# Read text from file
with open('input_text_001.txt', 'r') as file:
    text = file.read()

# Create an instance of TextProcessor
processor = TextProcessor(text)

# Tokenize the text into sentences
sentences = processor.tokenize(text)

# Process the first and second sentences
input_text = sentences[0]
output_text = sentences[1]

# Process the input sentence
input_array, attention_mask_array, position_array = processor.process_sentence(input_text)

# Process the output sentence
output_array, _, _ = processor.process_sentence(output_text)

# Decode the input and output arrays back to strings
input_string = processor.decode(input_array)
output_string = processor.decode(output_array)

# Print the results
print("Input array:", input_array)
print("attention_mask_array:", attention_mask_array)
print("position_array:", position_array)
print("Input string:", input_string)
print("Output array:", output_array)
print("Output string:", output_string)

# Output
# Input array: [30, 4, 201, 242, 29, 4, 158, 44, 289, 199, 24, 165, 130, 68, 280, 156, 4, 206, 130, 244, 58, 257, 228, 44, 107, 4, 155, 280, 122, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# attention_mask_array: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# position_array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# Input string: React.js , or simply React , is a widely-used open-source JavaScript library for building user interfaces , particularly for single-page applications that require a dynamic , interactive user experience .
# Output array: [13, 69, 16, 4, 29, 114, 100, 263, 90, 287, 58, 257, 70, 269, 54, 224, 112, 62, 92, 74, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Output string: Developed by Facebook , React enables developers to create web applications that can update and render efficiently as data changes .
