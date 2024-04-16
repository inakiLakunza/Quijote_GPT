import sys
import os

import pickle
import torch

from transformer_decoder import TransformerDecoder



# Encoder: take a string, output a list of integers
encode = lambda s: [s2i[c] for c in s]
# Decoder: Take a list of integers, output a string
decode = lambda l: "".join([i2s[i] for i in l])


if __name__ == '__main__':

    MODEL_PATH = '/ghome/group07/example2/quijote_GPT/saved_models/tiny_shakespeare/iters500000_bestval.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # HYPERPARAMETERS
    #==================================================
    BATCH_SIZE = 64
    BLOCK_SIZE = 256
    MAX_ITERS = 50000
    EVAL_INTERVAL = 500
    LEARNING_RATE = 3e-4 # self attention needs to have a quite low lr
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EVAL_ITERS = 200
    TRAIN_SPLIT_PERCENT = 90

    # Number of embedding dimensions
    N_EMBD = 384

    N_HEADS = 6  # --> HEAD SIZE will be N_EMBD // N_HEADS = 64

    N_LAYERS = 6
    DROPOUT = 0.2
    #==================================================





    # Generating str and int encoder and decoder
    #====================================================
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o tiny_shakespeare.txt
    with open('./databases/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    VOCAB_SIZE = len(chars)

    s2i = {ch:i for i,ch in enumerate(chars)}
    i2s = {i:ch for i, ch in enumerate(chars)}

    # Encoder: take a string, output a list of integers
    encode = lambda s: [s2i[c] for c in s]
    # Decoder: Take a list of integers, output a string
    decode = lambda l: "".join([i2s[i] for i in l])
    #====================================================

    model = TransformerDecoder(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, N_HEADS, 
                               N_LAYERS, device=DEVICE, dropout=DROPOUT)
    
    print(f"Device we are working on: {DEVICE}")
    if DEVICE == "cpu":
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))

    m = model.to(DEVICE)

    # generate from the model
    print("\n\n\nNOW SOME GENERATION WILL BE WRITTEN, UP TO 10000 TOKENS...")
    print("Creating context...")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print("Context created, generating output...:\n\n")
    print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))



