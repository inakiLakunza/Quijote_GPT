
import torch
import torch.nn as nn
from torch.nn import functional as F

# HYPERPARAMETERS
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
TRAIN_SPLIT_PERCENT = 90

# Number of embedding dimensions
N_EMBD = 32


torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o tiny_shakespeare.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

s2i = {ch:i for i,ch in enumerate(chars)}
i2s = {i:ch for i, ch in enumerate(chars)}

# Encoder: take a string, output a list of integers
encode = lambda s: [s2i[c] for c in s]
# Decoder: Take a list of integers, output a string
decode = lambda l: "".join([i2s[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(TRAIN_SPLIT_PERCENT/100*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BATCH_SIZE+1] for i in ix])
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
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
        out[split] = losses.mean()
    model.train()
    return out



class BigramLanguageModel(nn.Module):
    
    # We do not need to initialize our Bigram model with the vocab
    # size since it is already defined as a global variable
    def __init__(self):
        super().__init__()
        # n_embd == Number of Embedding Dimensions
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)

        # We will also encode the position of each token in the input
        # this way we will be able to know the position of the token
        # related to each embedding, since the position of each token
        # is important in a phrase.
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        # Language Modelling Head
        self.lm_head = nn.linear(N_EMBD, vocab_size)
        
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
        pos_embd = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C)
        
        # IN A BIGRAM MODEL THE POSITION OF EACH TOKEN DOES NOT MATTER SINCE
        # WE ARE JUST USING THE PREVIOUS TOKEN TO PREDICT THE FUTURE TOKEN
        # BUT WHEN WE ANALYZE CONTEXT WHICH IS FURTHER, THE LOCATION OF
        # EACH ONE OF THEM IS OF EXTREME IMPORTANCE

        # and so, now we will use both, the embedding of the value of
        # each token and the embedding of the position of each token
        # since both the value and the location are important information
        x = token_embd + pos_embd
        logits = self.lm_head(x)            # (B, T, vocab_size)
        
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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # logits becomes (B, C)
            # apply Softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx  
    



model = BigramLanguageModel(vocab_size)
m = model.to(DEVICE)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(MAX_ITERS):

    # every once in a while evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {iter}:\nTrain loss: {losses['train']:.4f}\nVal loss: {losses['val']:.4f}\n")
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))