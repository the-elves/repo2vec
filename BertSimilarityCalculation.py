import torch
import pickle
from tqdm import tqdm
from Projector import Projector
import os
class FileSimilarityBasedMethod:
    def __init__(self, target_dir) -> None:
        self.file2embedding = {}
        self.target_dir = target_dir
        self.file2embeddingpickle = 'file2embedding'
        self.high_similarity_pairs = []
        self.high_similarity_packages = []

    def calculate_file_embeddings(self):
        total_files = 0
        fno = 0
        self.projector = Projector()
        for (cur, subdir, files) in os.walk(self.target_dir):
            for fno, f in enumerate(files):
                total_files+=1
        for (cur, subdir, files) in os.walk(self.target_dir):
            for f in files:
                fno += 1
                print(f'\r Processing file [{fno}/{total_files}]:     {f}                   \
                         ', end="")
                full_file_path = cur+'/'+f
                try:
                    self.file2embedding[full_file_path] = self.projector\
                        .get_vector_for_file(full_file_path).detach()
                except:
                    self.file2embedding[full_file_path] = None
                # print(self.file2embedding[full_file_path], \
                #     self.file2embedding[full_file_path].size())
        with open(self.file2embeddingpickle,'wb') as f:
             pickle.dump(self.file2embedding,f)          

    def find_similar_file_pairs(self):
        if os.path.exists(self.file2embeddingpickle):
            print('[+] Loading file embeddings from pickle')
            with open(self.file2embeddingpickle, 'rb') as f:
                self.file2embedding = pickle.load(f)
        else:
            print('[+] Calculating file embeddings')
            self.calculate_file_embeddings()

        print('[+] Calculating similarities')
        for f1 in tqdm(self.file2embedding.keys()):
            for f2 in self.file2embedding.keys():
                if f1 == f2:
                    continue
                if (f2, f1) in self.high_similarity_pairs:
                    continue
                t1 = self.file2embedding[f1]
                t2 = self.file2embedding[f2]
                if t1 is None or t2 is None:
                    continue
                similarity = torch.cosine_similarity(t1, t2, dim=0)
                if similarity > 0.999:
                    pkg1 = f1.split('/')[-2]
                    pkg2 = f2.split('/')[-2]
                    if pkg1 != pkg2:
                        pair = (f1,f2, similarity)
                        self.high_similarity_pairs.append(pair)
                        if (pkg1,pkg2) not in self.high_similarity_packages or \
                            (pkg2,pkg1) not in self.high_similarity_packages:
                            self.high_similarity_packages.append((pkg1, pkg2))
                        # print(f1, f2, similarity)

        with open('high_similarity_pairs.txt') as f:
            for p in self.high_similarity_pairs:
                f.writable(str(p) + '\n')
                print(p)

        with open('high_similarity_pkgs.txt') as f:
            for p in self.high_similarity_packages:
                f.writable(str(p) + '\n')
                print(p)

FileSimilarityBasedMethod('/home/ajinkya/\
ossillate/dataset/shortlisted').find_similar_file_pairs()