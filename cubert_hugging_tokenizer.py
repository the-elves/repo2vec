from transformers import BertTokenizer
from utils import load_vocab, combine_tokenizer_with_subword
from tensor2tensor.data_generators import text_encoder
import os
from pathlib import Path
import collections
from cubert.python_tokenizer import PythonTokenizer


class CuBertHugTokenizer(BertTokenizer):
    # A hacky version that seems to work at least for python
    def __init__(
        self,
        vocab_file: Path
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=False,
            do_basic_tokenize=True,
            unk_token="[UNK]_",
            sep_token="[SEP]_",
            pad_token="<pad>_",
            cls_token="[CLS]_",
            mask_token="[MASK]_"
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.first_tokenizer = PythonTokenizer(512)
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(str(vocab_file))
        self._combined_func = combine_tokenizer_with_subword(
            self.first_tokenizer, self.subword_tokenizer)

    @property
    def do_lower_case(self):
        return False

    def _tokenize(self, text):
        return self._combined_func(text)

    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError

    def _convert_token_to_id(self, token):
        return self.subword_tokenizer._subtoken_string_to_id[token]