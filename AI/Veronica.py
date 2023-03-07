import torch
import torch.nn as nn
import torch.nn.functional as F
from AI  import loadData, Tokenizer

text = Tokenizer.get_raw_data()
chars = Tokenizer.get_tokens()
vocab_size = Tokenizer.get_vocab_size()
data = torch.tensor(Tokenizer.encode(text), dtype=torch.long)

n = int(0.8 * len(data)) #Takes 90% of data
train_data = data[:n] #Uses the 90% of Data as Training Value
val_data = data[n:] #Rest 10% of data as validation Data




#_____Hyperparameters______
batch_size = 16 
block_size = 32
max_iters = 5000
eval_interval = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
num_embd = 10
head_size = 10
num_heads = 10
n_layer = 10
#__________________________

x = train_data[:block_size]
y = train_data[1:block_size+1]


class Head(nn.Module ):
    def __init__(self , head_size):
        super().__init__()
        self.key = nn.Linear(num_embd, head_size, bias=False)
        self.query = nn.Linear(num_embd, head_size, bias=False)
        self.value = nn.Linear(num_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out 

class MultiHeadAttention(nn.Module ):
    def __init__(self , num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_embd)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, num_embd ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embd, 4 * num_embd),
            nn.ReLU(),
            nn.Linear(4 * num_embd, num_embd),
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, num_embd, n_heads) -> None:
        super().__init__()
        headsize = num_embd // n_heads
        self.sa = MultiHeadAttention(num_heads , headsize)
        self.ffwd = FeedForward(num_embd)
        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)
    def forward(self , x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class VeronicaModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, num_embd)
        self.position_embedding_table= nn.Embedding(block_size, num_embd)
        self.blocks = nn.Sequential(*[Block(num_embd, num_heads) for b in range(n_layer)])
        self.ln_f = nn.LayerNorm(num_embd)
        self.lm_head = nn.Linear(num_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[: , -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
 
model = VeronicaModel(vocab_size)

def get_model():
    return model

def get_eval_data():
    return eval_iters, max_iters , eval_interval

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def get_block_size():
    return block_size

def get_batch_size():
    return batch_size

def get_train_data():
    return train_data

def get_val_data():
    return val_data