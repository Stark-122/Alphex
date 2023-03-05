import torch 
#Reading Data from given DataSet
with open('DataSet/tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset in characters: {len(text)}")
print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#Tokenization
stoi = { ch:i for i , ch in enumerate(chars) }
itos = { i:ch for i , ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] #Encoder : Takes string input, gives integers and output
decode = lambda l: ''.join([itos[i] for i in l]) #Decoder : Takes integer input, and returns string output

print(encode("Hello there"))
print(decode([20, 43, 50, 50, 53, 1, 58, 46, 43, 56, 43]))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n= int(0.9 * len(data)) #Takes 90% of data
train_data = data[:n] #Uses the 90% of Data as Training Value
val_data = data[n:] #Rest 10% of data as valuation Data