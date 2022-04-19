from transformers import BertModel, BertConfig, BertTokenizer
from cubert_hugging_tokenizer import CuBertHugTokenizer
from typing import List
from utils import *
import os

class Projector:
    def __init__(self):
        data_dir = "/home/ajinkya/ossillate/repo2vec/cubert-pytorch"
        self.vocab = data_dir+'/github_python_minus_ethpy150open_deduplicated_vocabulary.txt'
        self.config_file = data_dir+'/bert_config.json'
        self.model_file = data_dir+'/torch-model'
        cubert_config = BertConfig.from_pretrained(self.config_file)
        self.tokenizer = CuBertHugTokenizer(self.vocab)
        self.model = BertModel.from_pretrained(os.fspath(self.model_file),
                                  config=cubert_config,
                                  local_files_only=True)
        self.MAX_LEN = 512
    
    def tokenize_file(self, filename):
        with open(filename) as f:
            source = f.read()
        toks = self.tokenizer(source, return_tensors="pt")
        for k in toks.keys():
            if toks[k].size()[1] > 512:
                toks[k] = toks[k][:,:512]
        return toks
                
                

    

    def get_vector_for_file(self, filename):
        tokens = self.tokenize_file(filename)
        outputs = self.model(**tokens)
        last_hidden_layer = outputs.last_hidden_state
        size = last_hidden_layer.size()[1]
        file_vec = last_hidden_layer[0][size-1]
        return file_vec


        
    

