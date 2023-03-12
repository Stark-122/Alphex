import re

with open('DataSet/tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
        input_text = f.read()
# Step 1: Read and preprocess input text
#input_text = "This is a sample text. It contains words and punctuations such as !, ?, and ."
#input_text = re.sub(r'[^\w\s\-\'_]', '', input_text)  # remove punctuations except hyphens and apostrophes

# Step 2: Define regular expression pattern to match words
pattern = r"\w+|[^\w\s]|\n"
  # match words with alphabets, digits, hyphens, apostrophes

# Step 3: Extract all the words from the input text using the pattern
words = re.findall(pattern, input_text)

# Step 4: Create vocabulary of unique words
vocab = {}
for word in words:
    if word not in vocab:
        vocab[word] = len(vocab)

# Step 5: Assign unique integer index to each word in the vocabulary
# index 0 will be reserved for padding, index 1 for unknown words
vocab_size = len(vocab) + 3
word_to_index = {word: index+3 for index, word in enumerate(vocab)}

def encode(text):
        #Encode input text by replacing each word with its corresponding integer index
        encoded_text = [word_to_index.get(word, 1) for word in words]  # 1 for unknown words
        return encoded_text

#Decode encoded text back to the original text
def decode(data):
        decoded_text = ' '.join([list(vocab.keys())[list(vocab.values()).index(index - 3)] for index in data])
        return decoded_text


def get_data():
      return input_text
def get_vocab_size():
      return vocab_size
