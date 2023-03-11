import os
import torch
import joblib
from AI import  Alphex, Tokenizer, Train





if __name__ == "__main__":
    if(os.path.exists("Model/AlphexLanguageModel.lm") == False):
        model = Alphex.AlphexLanguageModel()
        xb, yb = Alphex.get_batch('train')

        for b in range(Alphex.get_batch_size()):
            for t in range(Alphex.get_block_size()):
                context = xb[b, :t+1]
                target = yb[b, t]
        print(f"Model contains : {sum(p.numel() for p in model.parameters())//1e-6}  parameters.")
        logits, loss = model(xb , yb)
        print("training model...")
        Train.train_model()
        print(Tokenizer.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=2000)[0].tolist()))
    else:
        model_loaded = torch.load("Model/AlphexLanguageModel.lm") 
        print(f"Model contains : {sum(p.numel() for p in model_loaded.parameters())//1e-6}  parameters.")
        print("Model's output : ")
        print(Tokenizer.decode(model_loaded.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=2000)[0].tolist()))
