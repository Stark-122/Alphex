import torch 
#Reading Data from given DataSet
with open('DataSet/tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset in characters: {len(text)}")

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#Tokenization
stoi = { ch:i for i , ch in enumerate(chars) }
itos = { i:ch for i , ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] #Encoder : Takes string input, gives integers and output
decode = lambda l: ''.join([itos[i] for i in l]) #Decoder : Takes integer input, and returns string output


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n= int(0.9 * len(data)) #Takes 90% of data
train_data = data[:n] #Uses the 90% of Data as Training Value
val_data = data[n:] #Rest 10% of data as valuation Data

#Preparing the Data to Load into Neural Network
block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When context is {context}, target is {target}.")

torch.manual_seed(1337)
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x , y

xb , yb = get_batch('train')
print(f'Inputs: {xb.shape} , {xb}')
print (f'Targets: {yb.shape}, {yb}')
print('_________________')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b , t]
        print(f"When input is {context.tolist()} , target is {target}")