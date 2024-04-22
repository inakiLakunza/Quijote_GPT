
import typing

import regex as re

from utils import get_stats, merge
from utils import count_total_tokens    



class BasicTokenizer:
    """
    Implementation of a simple Tokenizer, which will be used for training
    it on a chosen text and then visualize the merged tokens.
    """

    def __init__(self):

        # Used for saving the merges we will do when training
        self.merges: dict[tuple[int, int], int]


        # Used to see if the Tokenizer has been trained or not
        # and if it has, we will be able to return the train text name
        self.trained: bool = False
        self.train_text_name: str = None
        self.vocab_size: int = None

        # Our vocabulary, it will be initialized with 
        # 256 elements (0 to 256), the elements of UTF-8
        self.vocab: dict[int, tuple[int, int]] = self._get_vocab()


    def _get_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        if self.trained:
            # if trained, merges will not be empty
            for (p0, p1), idx in self.merges.items():
                vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def train(self, text: str, vocab_size: int, verbose=True, train_text_name: str="No name specified"):
        """
        Train tokenizer with the given input text, and until 
        we get the wnated vocab size
        """


        assert vocab_size >= 256, "The new vocab size must be equal or larger than 256"

        # convert input txt into integers
        tokens = text.encode("utf-8")   # raw bytes
        tokens = list(map(int, tokens)) # conver to list of integers 0-255

        n_merges = vocab_size - 256
        ids: list[int] = list(tokens) # copy so we don't destroy the original list

        merges: dict[tuple[int, int], int] = {}
        for i in range(n_merges):
            stats = get_stats(ids)
            max_pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose: print(f"Merging {max_pair} into a new token: {idx}")
            ids = merge(ids, max_pair, idx)
            merges[max_pair] = idx

        if verbose:
            print("\n---------------------------\n") 
            print("tokens length: ", len(tokens))
            print("ids length: ", len(ids))
            print(f"Achieve compression ratio using {n_merges} merges: {len(tokens) / len(ids):.2f}X")

        # Set the trained info, if it was previously trained
        # the stored information will be lost
        self.trained = True
        self.vocab_size = vocab_size
        self.train_text_name = train_text_name

        self.merges = merges



    def encode(self, text: str) -> list[int]:
        """
        Given a string, return a list of integers (the tokens)
        """

        ids = list(text.encode("utf-8"))

        # If we do not have trained our Tokenizer
        # just return the UTF-8 encoding
        if self.trained:
            while len(ids) >= 2:
                stats = get_stats(ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break # nothing else can be merged
                idx = self.merges[pair]
                ids = merge(ids, pair, idx)
        
        return ids
        

    def decode(self, ids: list[int]) -> str:
        """
        Given ids (list of integers), return Python string
        """
        vocab = self._get_vocab()
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")

        return text
    




class RegexTokenizer(BasicTokenizer):
    """
    Regex Tokenizer, using the split pattern from GPT-4,
    and now there is the chance of using special tokens
    """

    def __init__(self, pattern=None):
        
        # Use GPT4's split pattern, else the introduced one 
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.split_pattern = pattern if pattern else GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.split_pattern)

        # Possibility of using special tokens
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        # init inheritance here, because we use special tokens now 
        # in _get_vocab(), so we first have to create it
        super().__init__()


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

        for k, v in special_tokens.items():
            if k in self.special_tokens.keys():
                # Delete previously registered item from the inverse dict
                self.inverse_special_tokens.pop(self.special_tokens[k])

            self.special_tokens[k] = v
            self.inverse_special_tokens[v] = k

    

    # @override
    def train(self, text: str, vocab_size: int, 
              verbose=True, train_text_name: str="No name specified",
              printfile: typing.IO=None):
        """
        Train tokenizer with the given input text, and until 
        we get the wanted vocab size
        If verbose specified, if printfile is None, it will print in the
        terminal, otherwise it will write the text in the specified 
        file (NORMAL WRITTING, NOT BINARY)
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
            if verbose: 
                if printfile is None:
                    print(f"Merging {max_pair} into a new token: {idx}")
                else:
                    printfile.write(f"Merging {max_pair} into a new token: {idx}\n")
            
            # make merges in the chunks
            ids = [merge(chunk_ids, max_pair, idx) for chunk_ids in ids]
            merges[max_pair] = idx

        if verbose:
            n_tokens_before = count_total_tokens(tokens)
            n_tokens_after = count_total_tokens(ids)

            if not printfile:
                print("\n---------------------------\n") 
                
                print("tokens length: ", n_tokens_before)
                print("ids length: ", n_tokens_after)
                print(f"Achieve compression ratio using {n_merges} merges: {(n_tokens_before / n_tokens_after):.2f}X")
            else:
                printfile.write("\n---------------------------\n")
                printfile.write(f"tokens length: {n_tokens_before}\n")
                printfile.write(f"ids length: {n_tokens_after}\n")
                printfile.write(f"Achieve compression ratio using {n_merges} merges: {(n_tokens_before / n_tokens_after):.2f}X\n")
            

        # Set the trained info, if it was previously trained
        # the stored information will be lost
        self.trained = True
        self.vocab_size = vocab_size
        self.train_text_name = train_text_name

        self.merges = merges



    def decode(self, ids: list[int]) -> str:
        """
        Given ids (list of integers), return Python string
        All the ids will be given in a single list,
        not separated in chunks. We do not need to use
        chunks now, since we are just decoding, we already
        have our vocab 
        """

        vocab = self._get_vocab()

        decoded_bytes = []

        # First check in vocab, since it will be the usual,
        # and then if it is an special token (which is less common)
        #so we do not have to go through two ifs
        for idx in ids:
            if idx in vocab:
                decoded_bytes.append(vocab[idx])
            # Check if idx belongs to a special character
            elif idx in self.inverse_special_tokens:
                decoded_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"{idx} does not refer to a valid token id")
            
        # Join bytes to form a string with them
        text_in_bytes = b"".join(decoded_bytes)
        # Decode byte string
        decoded_str = text_in_bytes.decode("utf-8", errors="replace")
        return decoded_str
    


    def _encode_chunk(self, chunk_in_bytes: list[bytes]) -> list[int]:
        """
        Encode chunk of bytes into ids 
        """

        # If we do not have trained our Tokenizer
        # just return the UTF-8 encoding
        ids = list(chunk_in_bytes)
        if self.trained:
            while len(ids) > 2:
                stats = get_stats(ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break # nothing else can be merged
                idx = self.merges[pair]
                ids = merge(ids, pair, idx)
        
        return ids



    def _encode_without_specials(self, text: str) -> list[int]:
        """
        Encoding without considering special tokens
        """

        # Split given text into chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        # Encode each chunk separately, and then all joined in ids list
        ids = []
        for text_chunk in text_chunks:
            chunk = text_chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk)
            ids.extend(chunk_ids)

        return ids



    def encode(self, text: str, allowed_special="none_raise") -> list[int]:
        """
        Given a string, return a list of integers (the tokens)
        We have to handle special tokens, and work in chunks

        We will choose which special tokens we allow
        -if 'none_raise', it means that we will not allow any, 
        and if we find one special token we will raise an error.
        -if 'all' , we will work with all the special tokens we have registered
        -if 'none', we will work with none, and not raise any error
        """

        special = None
        if allowed_special.lower() == "none":
            special = {}
        elif allowed_special.lower() == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif allowed_special.lower() == "all":
            special = self.special_tokens


        if not special:
            # just work without special tokens, much easier
            ids = self._encode_without_specials
            return ids
        
        # If we allow special tokens we have to search for them first
        # We have to split the text if we find any special token
        # We use re.split for this purpose
        # Note that if we surround the pattern with () we will
        # convert it into a capturing group, so we will include 
        # the special token
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        # Now we have separated all the special characters, and so
        # we can encode each group separately and then join the results
        ids = []
        for chunk in special_chunks:
            # See if the chunk contains a special token
            if chunk in special:
                ids.append(special[chunk])
            # Else we have an ordinary chunk
            else:
                extension = self._encode_without_specials(chunk)
                ids.extend(extension)
        
        return ids






def main() -> None:

    regex_tokenizer = RegexTokenizer()

    SPECIAL_TOKENS: dict[str, int] = {
            "<|initoftext|>": 500,
            "<|endoftext|>":  501,
            "<|initofbig|>": 510,
            "<|endofbig|>": 511,
            "<|dialogueinit|>": 520,
            "<|dialogueend|>": 521,
        }
    regex_tokenizer.register_special_tokens(SPECIAL_TOKENS)


    outfile = './output_regex_tokenizer_ancient_rome_with_special_tokens.txt'
    with open(outfile, 'w') as writef:

        # Try basic tokenizer using Tiny-Shakespeare
        with open('./../databases/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
            train_text = f.read()

        
        writef.write("Training regex tokeinzer...\n")
        regex_tokenizer.train(train_text, 276,
                              verbose=True, 
                              train_text_name="Tiny Shakespeare",
                              printfile=writef)
        
        with open('./../databases/ancient_rome_with_specials.txt', 'r', encoding='utf-8') as f:
            test_text = f.read()


        

        writef.write("\n\n\n----------------\n")
        writef.write("Encoding 'Ancient Rome' USING SPECIAL CHARACTERS:\n\n")
        test_text_encoded = regex_tokenizer.encode(test_text,
                                                   allowed_special="none")
        writef.write(str(test_text_encoded))


        writef.write("\n\n\n----------------\n")
        writef.write("Decoding the encoding of 'Ancient Rome':\n\n")
        test_text_decoded = regex_tokenizer.decode(test_text_encoded)
        writef.write(test_text_decoded)



if __name__ == '__main__':

    main()



