import gc
import os
import csv
import pickle

import spacy
# seems not needed but without it the program failed to find something
import scispacy
from scispacy.linking import EntityLinker

csv.field_size_limit(2147483647)


en_core_sci_sm_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz"
try:
    nlp = spacy.load("en_core_sci_sm")
except:
    print("downloading en_core_sci_sm...")
    import pip
    pip.main(["install", en_core_sci_sm_url])
    nlp = spacy.load("en_core_sci_sm")

nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "mesh"})
linker = nlp.get_pipe("scispacy_linker")

class TextDataset:

    def __init__(self, doc_text_file, ehr_entity_file):
        self.doc_texts = pickle.load(open(doc_text_file, 'rb'))
        self.ehr_entities = pickle.load(open(ehr_entity_file, 'rb'))
        self.all_doc_ids = list(self.doc_texts.keys())
        self.query_texts = {}
        self.texts = {}
        self.data = {}

    def add_ehr_text(self, filepath, ids2keep=None):
        reader = csv.reader(open(filepath))
        next(reader, None)
        if ids2keep is not None:
            ids2keep = pickle.load(open(ids2keep, 'rb'))
        for row in reader:
            if ids2keep is not None:
                if row[0] not in ids2keep:
                    continue
            self.query_texts[row[0]] = row[1]

    def create_dataset(self, outcome, censor_after_year):
        data = {}

        aid = [x for x in self.all_doc_ids if not x.startswith('PMC')]
        for id in aid:
            if int(self.doc_texts[id]['year']) > censor_after_year or self.doc_texts[id]['text'] == '': # why censoring > 2016?
                continue
            self.texts[id] = self.doc_texts[id]['text']
        del self.doc_texts
        gc.collect()

        for qid in self.query_texts.keys():
            data[qid] = {'outcome': outcome}
        self.data = data

    def add_entities(self, cutoff=0.9):
        for file in self.ehr_entities:
            if file not in self.data:
                print(f"file {file} not in self.data")
                continue
            entities = {'mesh': [], 'non-mesh': []}
            for sent in self.ehr_entities[file]:
                for entity in self.ehr_entities[file][sent]:
                    if 'mesh_ids' not in entity:
                        entities['non-mesh'].append(entity['mention'])
                        continue
                    for pair in entity['mesh_ids']:
                        term, prob = pair
                        if prob < cutoff:
                            continue
                        entity_mesh_text = linker.kb.cui_to_entity[term][1] if term in linker.kb.cui_to_entity else ''
                        if entity_mesh_text != '':
                            entities['mesh'].append(entity_mesh_text)
            self.data[file]['entities'] = entities

        del self.ehr_entities
        gc.collect()


class TRECDataset:

    def __init__(self, base_dir, years=None):
        train_file = os.path.join(base_dir, 'train-split-filtered.csv')
        dev_file = os.path.join(base_dir, 'dev-split-filtered.csv')
        test_file = os.path.join(base_dir, 'test-split-filtered.csv')

        self.train_data, train_texts = self.read_file(train_file, years)
        self.dev_data, dev_texts = self.read_file(dev_file, years)
        self.test_data, test_texts = self.read_file(test_file, years)
        self.texts = {**train_texts, **dev_texts, **test_texts}

    def read_file(self, filepath, years=None):
        reader = csv.reader(open(filepath))
        next(reader, None)
        data = {}
        texts = {}
        for row in reader:
            if row[0] not in data:
                if years is not None:   # Year-based filtering out of data
                    query_year = int(row[0][1:5])
                    if query_year not in years:
                        continue
                data[row[0]] = {'outcome': row[1], 'articles': {}}
                texts[row[0]] = 'What is the {}? '.format(row[1])+row[2]
            texts[row[3]] = row[4]
            data[row[0]]['articles'][row[3]] = {'judgement': 1 if int(row[5]) >=1 else -1}  # Categories 1 and 2 count as relevant
        return data, texts

    def add_entities(self, entity_file, cutoff=0.9):
        entity_file = pickle.load(open(entity_file, 'rb'))
        data_files = list(self.train_data.keys()) + list(self.dev_data.keys()) + list(self.test_data.keys())
        for file in entity_file:
            if file not in data_files:
                continue
            entities = {'mesh': [], 'non-mesh': []}
            for sent in entity_file[file]:
                for entity in entity_file[file][sent]:
                    if 'mesh_ids' not in entity:
                        entities['non-mesh'].append(entity['mention'])
                        continue
                    for pair in entity['mesh_ids']:
                        term, prob = pair
                        if prob < cutoff:
                            continue
                        entity_mesh_text = linker.kb.cui_to_entity[term][1] if term in linker.kb.cui_to_entity else ''
                        if entity_mesh_text != '':
                            entities['mesh'].append(entity_mesh_text)
            if file in self.train_data:
                self.train_data[file]['entities'] = entities
            elif file in self.dev_data:
                self.dev_data[file]['entities'] = entities
            else:
                self.test_data[file]['entities'] = entities
