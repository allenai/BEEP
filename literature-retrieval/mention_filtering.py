import os
import pickle
import csv
import spacy.cli
from nltk import word_tokenize, sent_tokenize
from spacy.tokens.doc import Doc
from spacy.tokens import Span
from medspacy.context import ConTextComponent, ConTextRule
import glob


"""
try:
    import en_core_web_sm
except:
    print('downloading "en_core_web_sm"')
    spacy.cli.download("en_core_web_sm")
    import en_core_web_sm
"""


print('downloading "en_core_web_sm"')
spacy.cli.download("en_core_web_sm")
import en_core_web_sm

import warnings

warnings.filterwarnings('ignore')


# Read in mentions and raw texts
def read_raw_text_files(file):
    files = {}
    if file.endswith(".csv"):
        reader = csv.reader(open(file))
        next(reader, None)
    elif file.endswith(".pkl"):
        reader = pickle.load(open(file, "rb"))
        reader = [(key, value["text"]) for key, value in reader.items()]
    else:
        reader = None
        raise (ValueError("file extension not recognized"))

    for row in reader:
        files[row[0]] = []
        text = row[1]
        sents = sent_tokenize(text)
        for sent in sents:
            words = word_tokenize(sent)
            files[row[0]].append(words)
    return files


def filter_mentions(mentions_file, text_file, outpath):
    print("loading en_core_web_sm")
    nlp = en_core_web_sm.load(disable=["tokenizer", "ner"])
    # spacy.load("en_core_web_sm", disable=["tokenizer","ner"])
    context = ConTextComponent(nlp, rules="default", use_context_window=True, max_scope=5)

    # Add paths to files containing extracted mentions and raw texts here
    # In our pipeline, mentions are extracted using a model trained on the i2b2 2010 dataset

    print(f"reading datasets: {os.path.basename(text_file)}, {os.path.basename(mentions_file)}")
    texts = read_raw_text_files(text_file)
    mentions = pickle.load(open(mentions_file, 'rb'))
    print("done.")

    # Apply ConText algorithm to identify and filter negated entities
    filtered_mentions = {}
    file_ind = 0
    for file in mentions: #texts:
        print('Processsing file {} ({})'.format(file, file_ind))
        file_ind += 1
        filtered_mentions[file] = {}
        file_lines = texts[file]
        file_mentions = mentions[file]
        assert len(file_lines) == len(file_mentions)

        for i, line in enumerate(file_lines):
            cur_ment = file_mentions[i]
            if not cur_ment:
                continue
            # If mentions are present in sentence, perform negation-based filtering
            filtered_mentions[file][i] = []
            doc = Doc(nlp.vocab, line)
            for name, proc in nlp.pipeline:
                doc = proc(doc)
            entities = []
            for mention in cur_ment:
                mention_words = mention["mention"]
                line_words = line[mention['start_offset']:(mention['end_offset'] + 1)]
                assert mention_words == line_words
                entities.append(Span(doc, mention['start_offset'], mention['end_offset'] + 1, mention['pred_type']))
            doc.ents = tuple(entities)
            doc = context(doc)

            for ent in doc.ents:
                if ent._.is_negated:
                    continue
                filtered_mentions[file][i].append(
                    {'mention': ent.text, 'start_offset': ent.start, 'end_offset': ent.end, 'pred_type': ent.label_})
        all_mention_count = sum([len(y) for x, y in mentions[file].items()])
        filtered_mention_count = sum([len(y) for x, y in filtered_mentions[file].items()])
        print('{} mentions kept out of {} for file {}'.format(filtered_mention_count, all_mention_count, file))

    pickle.dump(filtered_mentions, open(outpath, 'wb'))

outcome = "mortality"
mentions_files = glob.glob("./mortality_literature_mentions*.pkl")
print(mentions_files)
for mention_file in mentions_files:
    #mention_file = fr"../data/{outcome}/mentions/mortality_{split}_mentions.pkl"
    text_file = r"../data/outcome-literature/mortality_texts_and_dates.pkl"
    outpath = mention_file.replace("mentions", "filtered_mentions")
    filter_mentions(mention_file, text_file, outpath)


