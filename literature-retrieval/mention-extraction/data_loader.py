import os
import csv
import math
from nltk import sent_tokenize, word_tokenize
import pickle

class RawTextDataset:
    def __init__(self, file, tag_type):
        self.data = self.read_raw_text_files(file)
        self.label_vocab = {'O': 0, 'B-PROB': 1, 'I-PROB': 2, 'B-TREAT': 3, 'I-TREAT': 4, 'B-TEST': 5, 'I-TEST': 6} \
            if tag_type == 'i2b2' \
            else {'O': 0, 'I-PAR': 1, 'I-INT': 2, 'I-OUT': 3}


    def read_raw_text_files(self, file):
        files = {}
        if file.endswith(".csv"):
            reader = csv.reader(open(file))
            next(reader, None)
        elif file.endswith(".pkl"):
            reader = pickle.load(open(file, "rb"))
            reader = [(key, value["text"]) for key, value in reader.items()]
        else:
            reader = None
            raise(ValueError("file extension not recognized"))

        for row in reader:
            files[row[0]] = []
            text = row[1]
            sents = sent_tokenize(text)
            for sent in sents:
                words = word_tokenize(sent)
                files[row[0]].append(words)
        return files


class i2b2Dataset:

    def __init__(self, root_dir):

        self.train_data, self.dev_data = self.read_conll_format_file(os.path.join(root_dir, 'train.txt'), split=True)
        self.test_data = self.read_conll_format_file(os.path.join(root_dir, 'test.txt'))
        self.data = {"train": self.train_data, "dev": self.dev_data, "test": self.test_data}

        # Convert labels to integer values
        self.init_label_vocab()
        self.construct_label_sequences(self.train_data)
        self.construct_label_sequences(self.dev_data)
        self.construct_label_sequences(self.test_data)


    def read_conll_format_file(self, filepath, split=False):  # Split train.txt into train and dev data
        reader = open(filepath)
        ex_id = 0
        tokens = []
        pico = []
        examples = []   # Each example is a dict containing ID, tokens and NER labels

        for line in reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    examples.append({
                            'id': ex_id,
                            'tokens': tokens,
                            'labels': pico
                        })
                    ex_id += 1
                    tokens = []
                    pico = []
            else:
                ex_data = line.split()
                tokens.append(ex_data[0])
                pico.append(ex_data[-1].rstrip())
        # Last example may be left out
        if tokens:
            examples.append({
                    'id': ex_id,
                    'tokens': tokens,
                    'labels': pico
                })

        if split:
            dev_prop = math.ceil(0.1*len(examples))
            return examples[dev_prop:], examples[:dev_prop]
        return examples


    def init_label_vocab(self):
        self.label_vocab = {
                    'O': 0,
                    'B-PROB': 1,
                    'I-PROB': 2,
                    'B-TREAT': 3,
                    'I-TREAT': 4,
                    'B-TEST': 5,
                    'I-TEST': 6
                }


    def construct_label_sequences(self, examples):
        for example in examples:
            label_seq = [self.label_vocab[x] for x in example['labels']]
            example['gold_seq'] = label_seq


class PICODataset:

    def __init__(self, root_dir):

        self.train_data = self.read_conll_format_file(os.path.join(root_dir, 'train.txt'))
        self.dev_data = self.read_conll_format_file(os.path.join(root_dir, 'dev.txt'))
        self.test_data = self.read_conll_format_file(os.path.join(root_dir, 'test.txt'))
        self.data = {"train": self.train_data, "dev": self.dev_data, "test": self.test_data}

        # Convert labels to integer values
        self.init_label_vocab()
        self.construct_label_sequences(self.train_data)
        self.construct_label_sequences(self.dev_data)
        self.construct_label_sequences(self.test_data)


    def read_conll_format_file(self, filepath):
        reader = open(filepath)
        ex_id = 0
        tokens = []
        pico = []
        examples = []   # Each example is a dict containing ID, tokens and pico labels

        for line in reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    examples.append({
                            'id': ex_id,
                            'tokens': tokens,
                            'labels': pico
                        })
                    ex_id += 1
                    tokens = []
                    pico = []
            else:
                ex_data = line.split()
                tokens.append(ex_data[0])
                pico.append(ex_data[-1].rstrip())
        # Last example may be left out
        if tokens:
            examples.append({
                    'id': ex_id,
                    'tokens': tokens,
                    'labels': pico
                })

        return examples


    def init_label_vocab(self):
        self.label_vocab = {
                    'O': 0,
                    'I-PAR': 1,
                    'I-INT': 2,
                    'I-OUT': 3,
                }


    def construct_label_sequences(self, examples):
        for example in examples:
            label_seq = [self.label_vocab[x] for x in example['labels']]
            example['gold_seq'] = label_seq
