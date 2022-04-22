import csv
import os
import pickle
csv.field_size_limit(2147483647)
from collections import Counter

class EHRDataset:

    def __init__(self, train_path, dev_path, test_path):
        self.train_data = self.read_csv(train_path)
        self.dev_data = self.read_csv(dev_path)
        self.test_data = self.read_csv(test_path)

    def read_csv(self, path):
        reader = csv.reader(open(path))
        data = {}
        next(reader, None)
        for row in reader:
            data[row[0]] = {'ehr': row[1], 'outcome': int(row[2])}
        return data

    def compute_class_weights(self):
        class_counts = Counter()
        for example in self.train_data:
            class_counts.update([self.train_data[example]['outcome']])
        num_samples = sum(list(class_counts.values()))
        num_classes = len(class_counts)
        balance_coeff = float(num_samples)/num_classes
        self.class_weights = {k:balance_coeff/float(v) for k,v in class_counts.items()}

    def add_relevant_literature(self, lit_dir, topk, use_pico):
        all_texts = pickle.load(open('../data/pubmed_texts_and_dates.pkl', 'rb'))
        if use_pico:
            all_texts = pickle.load(open('../data/pico_highlighted_pubmed_texts.pkl', 'rb'))
        for file in os.listdir(lit_dir):
            id = file.split('.pkl')[0]
            docs = pickle.load(open(os.path.join(lit_dir, file), 'rb'))
            docs = docs[id]
            docs = list(reversed(sorted(docs, key=lambda x:x[1])))
            chosen_docs = docs[:int(topk)] if topk >= 1 else [x for x in docs if x[1] >= topk]
            chosen_docs = [[x[0], all_texts[x[0]]['text'], x[1]] for x in chosen_docs]  # We may want to include year later?
            if id in self.train_data:
                self.train_data[id]['pubmed_docs'] = chosen_docs
            elif id in self.dev_data:
                self.dev_data[id]['pubmed_docs'] = chosen_docs
            elif id in self.test_data:
                self.test_data[id]['pubmed_docs'] = chosen_docs

    def add_literature_matrices(self, lit_embed_file):
        lit_embeds = pickle.load(open(lit_embed_file, 'rb'))
        for id in self.train_data:
            self.train_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.train_data[id]['pubmed_docs']}
        for id in self.dev_data:
            self.dev_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.dev_data[id]['pubmed_docs']}
        for id in self.test_data:
            self.test_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.test_data[id]['pubmed_docs']}
