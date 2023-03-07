import os
from AI import loadData

#Reading Data from given DataSet
os.listdir()
with open('DataSet/tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Tokenization
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] #Encoder : Takes string input, gives integers and output
decode = lambda l: ''.join([itos[i] for i in l]) #Decoder : Takes integer input, and returns string output

def get_raw_data():
    return text

def get_tokens():
    return chars

def get_vocab_size():
    return vocab_size