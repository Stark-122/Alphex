import os
from tokenizers import Tokenizer
from tokenizers.models import BPE

#Reading Data from given DataSet
os.listdir()
with open('DataSet/tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
chars =  sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)

#Tokenization
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
EncodeData = lambda s: [stoi[c] for c in s] #Encoder : Takes string input, gives integers and output
DecodeData = lambda l: ''.join([itos[i] for i in l]) #Decoder : Takes integer input, and returns string output

def encode(data):
    return EncodeData(data)
def decode(data):
    return DecodeData(data)


def get_raw_data():
    return text

def get_tokens():
    return chars

def get_vocab_size():
    return vocab_size