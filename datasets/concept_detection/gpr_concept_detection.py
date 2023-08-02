import json
import os
import re

import torch
from torchtext.data import get_tokenizer

SOS = 'startofsentence'
EOS = 'endofsentence'


class GPRConceptsDataset:
    def __init__(self):
        if os.getenv("GPR_DATASET_CONCEPT_DETECTION") is None:
            raise RuntimeError('Dataset path must be set up.')
        root = os.environ['GPR_DATASET_CONCEPT_DETECTION']
        embs_dir = os.path.join(root, 'embs')
        self.descriptions_path = os.path.join(root, 'categories.json')
        self.embs_list = [os.path.join(embs_dir, file_name) for file_name in sorted(os.listdir(embs_dir))]

        # load descriptions
        with open(self.descriptions_path, 'r') as json_file:
            self.descriptions = json.load(json_file)

        for key, value in self.descriptions.items():
            self.descriptions[key] = re.sub(r"[^a-zA-Z0-9]+", ' ', value.lower())

        # get text corpus
        text_corpus = list(self.descriptions.values())

        # init tokenizer
        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        # tokenize text corpus
        self.tokenized_corpus = [self.tokenizer(text) for text in text_corpus]

        # get vocabulary
        self.vocab = [SOS, EOS]
        for sentence in self.tokenized_corpus:
            for token in sentence:
                if token not in self.vocab:
                    self.vocab.append(token)

    def __len__(self):
        return len(self.embs_list)

    def __getitem__(self, idx):
        emb = torch.load(self.embs_list[idx], map_location=torch.device('cpu'))
        emb_class = int(os.path.basename(self.embs_list[idx]).split('_')[0])
        description = self.descriptions[str(emb_class)]
        tokenized = self.tokenizer(f'{SOS} {description} {EOS}')
        tokenized_tensor = torch.tensor([self.vocab.index(token) for token in tokenized], dtype=torch.int64)
        return {'embs': emb, 'tokens': tokenized_tensor}
