

import regex as re

from basic_tokenizer import BasicTokenizer

from utils import get_stats, merge
from utils import count_total_tokens    

class RegexTokenizer(BasicTokenizer):
    """
    Regex Tokenizer, using the split pattern from GPT-4,
    and now there is the chance of using special tokens
    """


    def __init__(self, pattern=None):
        super().__init__()

        # Use GPT4's split pattern, else the introduced one 
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.split_pattern = pattern if pattern else GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.split_pattern)

        # Possibility of using special tokens
        self.special_tokens = {}
        self.inverse_special_tokens = {}



    # @override
    def _get_vocab(self):

        vocab = super()._get_vocab()

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab



    def register_special_tokens(self, special_tokens: dict[str, int]):
        """
        Example of special token:
        {"<|endoftext|>": 100257}
        
        If we want to re-register a token which we had registered
        previously it will be rewritten.

        IF WE KNOW WE WILL DO THIS PROCESS JUST ONCE AND INSERT ALL SPECIAL
        TOKENS AT ONCE, WE CAN OPTIMIZE THE CODE. WE CAN SIMPLY USE:
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        """

        for k, v in special_tokens:
            if k in self.special_tokens:
                # Delete previously registered item from the inverse dict
                self.inverse_special_tokens.pop(self.special_tokens[k])

            self.special_tokens[k] = v
            self.inverse_special_tokens[v] = k

    
    # @override
    def train(self, text: str, vocab_size: int, verbose=True, train_text_name: str="No name specified"):
        """
        Train tokenizer with the given input text, and until 
        we get the wnated vocab size
        """

        assert vocab_size >= 256, "The new vocab size must be equal or larger than 256"
        n_merges = vocab_size - 256

        # Split the text into chunks, using the chosen pattern
        text_chunks = re.findall(self.compiled_pattern, text)

        # convert text of chunks into integers
        tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        ids = tokens.copy() # copy so we don't destroy the original list

        merges: dict[tuple[int, int], int] = {}
        for i in range(n_merges):

            # The get stats method allows to load a dictionary,
            # so we can iterate through the chunks without 
            # loosing information
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, load_counts=stats)
            
            max_pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose: print(f"Merging {max_pair} into a new token: {idx}")
            
            # make merges in the chunks
            ids = [merge(chunk_ids, max_pair, idx) for chunk_ids in ids]
            merges[max_pair] = idx

        if verbose:
            n_tokens_before = count_total_tokens(tokens)
            n_tokens_after = count_total_tokens(ids)
            print("\n---------------------------\n") 
            
            print("tokens length: ", n_tokens_before)
            print("ids length: ", n_tokens_after)
            print(f"Achieve compression ratio using {n_merges} merges: {(n_tokens_before / n_tokens_after):.2f}X")


        # Set the trained info, if it was previously trained
        # the stored information will be lost
        self.trained = True
        self.vocab_size = vocab_size
        self.train_text_name = train_text_name

        self.merges = merges


