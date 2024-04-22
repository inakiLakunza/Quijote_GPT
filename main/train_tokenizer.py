
import os
import sys

import pickle

from Tokenizer import RegexTokenizer





def main() -> None:

    tokenizer = RegexTokenizer()

    WANTED_NEW_TOKENS: int = 200

    SPECIAL_TOKENS: dict[str, int] = {
            "<|beginchaptername|>": 1000,
            "<|endchaptername|>": 1001
        }
    
    outfile = './tokenizer_train_cien_años_de_soledad.txt'
    with open(outfile, 'w') as writef:

        writef.write("\nRegistering the following special tokens:\n")
        writef.write(f"{SPECIAL_TOKENS.items()}, \n\n")
        tokenizer.register_special_tokens(SPECIAL_TOKENS)

        # Try basic tokenizer using Tiny-Shakespeare
        with open('./../databases/cien_años_de_soledad.txt', 'r', encoding='utf-8') as f:
            train_text = f.read()

        
        writef.write("Training regex tokeinzer...\n")
        tokenizer.train(train_text, 256+WANTED_NEW_TOKENS,
                        verbose=True, 
                        train_text_name="Cien años de soledad",
                        printfile=writef)
        
        with open('./../databases/cien_años_de_soledad.txt', 'r', encoding='utf-8') as f:
            test_text = f.read()


        

        writef.write("\n\n\n----------------\n")
        writef.write("Encoding 'Cien años de soledad':\n\n")
        test_text_encoded = tokenizer.encode(test_text,
                                             allowed_special="all")
        writef.write(str(test_text_encoded))


        writef.write("\n\n\n----------------\n")
        writef.write("Decoding the encoding of 'Cien años de soledad':\n\n")
        test_text_decoded = tokenizer.decode(test_text_encoded)
        writef.write(test_text_decoded)


    SAVENAME = "pkl_tokenizer.pkl"
    print(f"\n\nTraining and try completed, saving tokenizer as {SAVENAME}")
    with open(SAVENAME, "wb") as f:
        pickle.dump(tokenizer, f)



if __name__ == "__main__":

    main()

