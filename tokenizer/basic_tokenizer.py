
from utils import get_stats, merge


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
        self.vocab_size: int

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
        

    def decode(self, ids: list[int]) :
        """
        Given ids (list of integers), return Python string
        """
        vocab = self._get_vocab()
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")

        return text
    


if __name__ == '__main__':

    outfile = './basic_tokenizer_ancient_rome_output.txt'
    with open(outfile, 'w') as writef:

        # Try basic tokenizer using Tiny-Shakespeare
        with open('./../databases/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
            train_text = f.read()

        basic_tokenizer = BasicTokenizer()
        
        writef.write("Training basic tokeinzer...\n")
        basic_tokenizer.train(train_text, 276,
                            verbose=True, train_text_name="Tiny Shakespeare")
        
        with open('./../databases/ancient_rome.txt', 'r', encoding='utf-8') as f:
            test_text = f.read()


        writef.write("\n\n\n----------------\n")
        writef.write("Encoding 'Ancient Rome':\n")
        test_text_encoded = basic_tokenizer.encode(test_text)
        writef.write(str(test_text_encoded))


        writef.write("\n\n\n----------------\n")
        writef.write("Decoding the encoding of 'Ancient Rome':\n")
        test_text_decoded = basic_tokenizer.decode(test_text_encoded)
        writef.write(test_text_decoded)