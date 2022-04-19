from cubert.python_tokenizer import PythonTokenizer
from cubert.cubert_tokenizer import CuBertTokenizer
import collections
from typing import List, Callable
from tensor2tensor.data_generators import text_encoder


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def combine_tokenizer_with_subword(
    initial_tokenizer: CuBertTokenizer,
    subword_tokenizer: text_encoder.SubwordTextEncoder
    ) -> Callable[[str], List[str]]:
    # Try to match the functionality at 
    # https://github.com/google-research/google-research/blob/50c6cd94b5/cubert/code_to_subtokenized_sentences.py#L111-L118
    def tokenize(string: str) -> List[str]:
        toks = initial_tokenizer.tokenize(string)
        return flatten_list(
            subword_tokenizer.decode_list(
                subword_tokenizer.encode_without_tokenizing(token)
            )
            for token in toks
        )
    return tokenize

def flatten_list(t):
    return [item for sublist in t for item in sublist]
