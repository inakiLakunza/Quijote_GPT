
import sys
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

import tqdm
import pickle

from building_blocks import Block, FeedForward
from Tokenizer import RegexTokenizer

import wandb


def write_generation_in_file(file):
    pass



def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# We have no Dropout, Batch Norm, ... laeyrs, but it is good
# practice to explain at each point in which mode our NN is in
# On the other hand, with @torch.no_grad() we tell PyTorch that
# when inside this function not to compute the gradients for then
# being able to make a step, since this is a evaluation process,
# not a train process, so it is more memory and time efficient
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out






class TransformerDecoder(nn.Module):
        
    # We do not need to initialize our model with the vocab
    # size since it is already defined as a global variable
    def __init__(self, vocab_size, n_embd, block_size, n_heads, n_layers, device, dropout=0.2):
        super().__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        # n_embd == Number of Embedding Dimensions
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)

        # We will also encode the position of each token in the input
        # this way we will be able to know the position of the token
        # related to each embedding, since the position of each token
        # is important in a phrase.
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        
        # INSTEAD OF JUST USING A SINGLE HEAD OF 32 DIMENSIONAL SELF ATTENTION
        # WE MUST ASSURE THAT N_EMBD=N_HEADS*HEAD_SIZE, BECAUSE WE MUST MAITAIN THE DIMENSIONALITY!! 
        self.blocks = nn.Sequential(*[Block(self.n_embd, self.block_size, n_heads=self.n_heads, dropout=self.dropout) for _ in range(self.n_layers)])

        self.ln_f = nn.LayerNorm(self.n_embd) # final layer norm
        
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape


        # idx and targets are both (B, T) tensor of integers
        
        # This is not going to give us directly the logits,
        # it will first give us the token embeddings

        # Now we will have two different channel numbers, the one
        # for the number of dimensions it has our embedding, and the
        # other one for the number of channels it has our logits, which
        # is our vocab size. So we will use C for the dimensions
        # and then we will write vocab size
        token_embd = self.token_embedding_table(idx) # (B, T, C)
        # the arange will just give us integers from position 0 to T-1
        pos_embd = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        
        # IN A BIGRAM MODEL THE POSITION OF EACH TOKEN DOES NOT MATTER SINCE
        # WE ARE JUST USING THE PREVIOUS TOKEN TO PREDICT THE FUTURE TOKEN
        # BUT WHEN WE ANALYZE CONTEXT WHICH IS FURTHER, THE LOCATION OF
        # EACH ONE OF THEM IS OF EXTREME IMPORTANCE

        # and so, now we will use both, the embedding of the value of
        # each token and the embedding of the position of each token
        # since both the value and the location are important information
        x = token_embd + pos_embd # (B, T, C)
        x = self.blocks(x)        # (B, T, C)
        x = self.ln_f(x)          # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    
    def generate(self, idx, end_chapter_token_idx = 459, max_new_tokens=15000):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # BECAUSE WE ARE USING POSITIONAL ENCODING, WE HAVE TO
            # MAKE SURE THAT OUR idx THAT WE FED INTO THE MODEL,
            # IS NEVER LARGER THAN THE BLOCK SIZE WE CHOOSE,
            # BECAUSE OTHERWISE THE POSITIONAL EMBEDDING TABLE
            # IS GOING TO RUN OUT OF SCOPE
            idx_cond = idx[:, -self.block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # logits becomes (B, C)
            # apply Softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
            # Break if created chapter has finished, will finish when
            # a new <|enchapter|> is created, which belongs to token id: 2001
            if (idx_next == end_chapter_token_idx).any().item():
                break

        return idx  
    




if __name__ == '__main__':


    # HYPERPARAMETERS
    #==================================================
    BATCH_SIZE = 64
    BLOCK_SIZE = 256
    MAX_ITERS = 100000
    EVAL_INTERVAL = 500
    LEARNING_RATE = 1e-3 # self attention needs to have a quite low lr
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EVAL_ITERS = 200

    # Number of embedding dimensions
    N_EMBD = 384

    N_HEADS = 6  # --> HEAD SIZE will be N_EMBD // N_HEADS = 64

    N_LAYERS = 6
    DROPOUT = 0.2
    #==================================================

    wandb.login(key="34db2c5ef8832f040bb5001755f4aa5b64cf78fa",
                relogin=True)

    wandb.init(
        project = "Quijote-GPT",
        name = "Tokenizer 200, 100000iters",
        config={
            "tokenizer": "Regex, 200 merges",
            "dataset": "QUIJOTE",
            "tokenizer train": "Cien años de soledad",
            "iters": MAX_ITERS,
            "manual_seed": 1337,
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "learning_rate": LEARNING_RATE,
        }
    )

    wandb.define_metric("loss", summary="min")
    wandb.define_metric("val_loss", summary="min")


    torch.manual_seed(1337)

    with open('./../databases/quijote.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # EXACT NUMBER SO WE CAN USE AS VALIDATION
    # THE LAST CHAPTER
    SPLIT_IN: int = -7296
    # All chapters except last for training
    # Last chapter for training
    train_text = text[:SPLIT_IN]
    val_text = text[SPLIT_IN:]

    chars = sorted(list(set(text)))


    # TOKENIZER
    #=============================================================
    # Load Tokenizer (trained with 'Cien años de soledad')
    with open("./pkl_tokenizer.pkl", 'rb') as f:
        tokenizer: RegexTokenizer = pickle.load(f)

    assert tokenizer.trained, "Loaded tokenizer is not trained"
    
    # FORGOT TO ADD THESE TWO SPECIAL TOKENS TO THE ONES WE ALREADY
    # HAVE, SO I CAN ADD THEM NOW, AND I WILL REMAP THE ADDED
    # TWO PREVIOUSLY, SINCE WE NEED TO ADD THE IDX VALUES
    # STARTING FROM THE LAST WE HAVE IN OUR VOCAB, BECAUSE
    # IN THE MODEL WE HAVE TO CREATE A LINEAR LAYER
    # FOR THE EMBEDDING
    tokenizer.register_special_tokens({
        "<|beginchaptername|>": 456,
        "<|endchaptername|>":   457,
        "<|beginchapter|>":     458,
        "<|endchapter|>":       459,
    })

    print("Special Tokens:\n", tokenizer.special_tokens)

    # Tokenize data
    train_data = torch.tensor(tokenizer.encode(train_text, allowed_special="all"), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text, allowed_special="all"), dtype=torch.long)
    
    # Get vocab size from tokenizer
    vocab_size = tokenizer.vocab_size
    vocab_size += 4
    print(f"Vocab size: {vocab_size}")
    assert vocab_size > 255, "Vocab size is lower than 256, something is wrong"
    #=============================================================


    model = TransformerDecoder(vocab_size, N_EMBD, BLOCK_SIZE, N_HEADS, 
                               N_LAYERS, device=DEVICE, dropout=DROPOUT)
    m = model.to(DEVICE)

    print("Number of parameters of the model:")
    print(f"{sum(p.numel() for p in m.parameters())/1e6} 'M parameters\n\n")

    wandb.watch(m, log_freq=100)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    min_val_loss = 1000000
    with open("generations_quijote_GPT_higher_lr_2.txt", "w") as writef:
        for iter in tqdm.tqdm(range(MAX_ITERS)):

            # every once in a while evaluate the loss on train and val sets
            if iter % EVAL_INTERVAL == 0:
                losses = estimate_loss()
                train_loss = losses['train']
                val_loss = losses['val']
                print(f"Step {iter}:\nTrain loss: {train_loss:.4f}\nVal loss: {val_loss:.4f}\n")
                wandb.log({"iter": iter+1, "loss": train_loss, "val_loss": val_loss})
            
                line1 = f"\n\n\n Predictions at iter {iter}:\n\n"
                writef.write(line1)
                context = torch.tensor([[458]], dtype=torch.long, device=DEVICE)
                line2 = tokenizer.decode(m.generate(context, max_new_tokens=5000)[0].tolist())
                writef.write(line2)
                writef.write("\n\n")

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_path = os.path.join("./", "saved_models", "quijote_GPT", )
                    best_val_save_path = os.path.join(save_path, "tokenizer200_100000iters.pth")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(m.state_dict(), best_val_save_path)

            
            # sample a batch of data
            xb, yb = get_batch('train')
            
            # evaluate the loss
            logits, loss = m(xb, yb)
            wandb.log({"iter": iter+1, "loss": loss})
            print(f"Step {iter}:\nTrain loss: {loss:.4f}\n")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    wandb.finish()


    # generate from the model
    print("""
          \n\n\nNOW SOME GENERATION WILL BE WRITTEN, UNTIL 15000 tokens
          are generated or until the end of chapter token (<|endchapter|>)is generated.
          """)
    context = torch.tensor([[458]], dtype=torch.long, device=DEVICE)
    print(tokenizer.decode(m.generate(context, max_new_tokens=15000)[0].tolist()))