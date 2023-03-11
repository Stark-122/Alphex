import os
import torch
import joblib
from AI import Alphex

model = Alphex.get_model()
eval_iters, max_iters , eval_interval = Alphex.get_eval_data()
checkpoint_dir = "Model/checkpoints"
best_train_loss = 0
best_val_loss = 0
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = Alphex.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out , loss


#training the model
def train_model():
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=Alphex.learning_rate)
    losses, loss = estimate_loss()
    best_train_loss = loss
    best_val_loss = loss
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses, loss = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = Alphex.get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    
    torch.save(model, "Model/VeronicaModel.lm")