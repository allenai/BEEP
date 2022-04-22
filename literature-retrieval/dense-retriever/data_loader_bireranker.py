import os
import csv
csv.field_size_limit(2147483647)
import pickle

import spacy
import scispacy
from scispacy.linking import EntityLinker

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "mesh"})
linker = nlp.get_pipe("scispacy_linker")

class TextDataset:

    def __init__(self, doc_id_file, doc_text_file, ehr_entity_file):
        self.all_doc_ids = pickle.load(open(doc_id_file, 'rb'))
        self.doc_texts = pickle.load(open(doc_text_file, 'rb'))
        self.ehr_entities = pickle.load(open(ehr_entity_file, 'rb'))

    def add_ehr_text(self, filepath, outcome):
        reader = csv.reader(open(filepath))
        next(reader, None)
        self.query_texts = {}
        self.doc_ids = {}
        ids2keep = pickle.load(open('../../data/pmv_ids.pkl', 'rb')) if outcome == 'pmv' else None
        for row in reader:
            if ids2keep is not None:
                if row[0] not in ids2keep:
                    continue
            self.query_texts[row[0]] = row[1]
            self.doc_ids[row[0]] = list(self.all_doc_ids)

    def create_dataset(self, outcome):
        data = {}
        self.texts = {}
        aid = [x for x in list(self.doc_ids.values())[0] if not x.startswith('PMC')]
        for id in aid:
            if int(self.doc_texts[id]['year']) > 2016 or self.doc_texts[id]['text'] == '':
                continue
            self.texts[id] = self.doc_texts[id]['text']
        for qid in self.doc_ids:
            data[qid] = {'outcome': outcome}
        self.data = data

    def add_entities(self, cutoff=0.9):
        for file in self.ehr_entities:
            if file not in self.data:
                continue
            # print('Adding entities for {}'.format(file))
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
