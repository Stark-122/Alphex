import os
import torch
import joblib
from AI import  Veronica, Tokenizer, Train





if __name__ == "__main__":
    if(os.path.exists("VeronicaModel.lm") == False):
        xb, yb = Veronica.get_batch('train')

        for b in range(Veronica.get_batch_size()):
            for t in range(Veronica.get_block_size()):
                context = xb[b, :t+1]
                target = yb[b, t]
                
        model = Veronica.VeronicaModel()
        print(f"Model contains : {sum(p.numel() for p in model.parameters())//1e-6}  parameters.")
        logits, loss = model(xb , yb)
        print("training model...")
        Train.train_model()
        
    else:
        model_loaded = torch.load("Model/VeronicaModel.lm") 
        print(f"Model contains : {sum(p.numel() for p in model_loaded.parameters())//1e-6}  parameters.")
        print(Tokenizer.decode(model_loaded.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=2000)[0].tolist()))
