'''
Code to dump NER or PICO tags for raw text. Model: Pretrained LM + linear layer
Run: python text_tagger.py --data <RAW TEXT CSV> --out_dir <OUTPUT DIR> --model_name_or_path <LM NAME> --checkpoint <MODEL WEIGHT FILE> --task <pico/i2b2>
'''

import pickle
import os
import argparse
import numpy as np

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, set_seed
from data_loader import RawTextDataset
from utils import extract_entities, batch_and_tokenize_data


def tag(model, data, label_vocab):
    with torch.no_grad():
        model.eval()
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        batch = {x: y.cuda() for x, y in data.items()}
        outputs = model(**batch)
        cur_preds = np.argmax(outputs[1].detach().cpu().numpy(), axis=2)
        labels = batch['labels'].cpu().numpy()
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(cur_preds, labels)
        ]
    return true_predictions


def run_tagger(data, out_dir, model_name_or_path, checkpoint, task, batch_size, outcome):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("building dataset (tokenizing sentence and words)...")
    dataset = RawTextDataset(data, task)
    label_vocab = dataset.label_vocab
    print("dataset built.\n")


    print("constructing config...")
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(list(label_vocab.keys())),
        label2id=label_vocab,
        id2label={i: l for l, i in label_vocab.items()},
        finetuning_task='ner',
        cache_dir='../../cache',
    )
    print("config constructed.\n")

    print("constructing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir='../../cache',
        use_fast=True,
    )
    print("tokenizer constructed.\n")

    print("constructing model...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir='../../cache',
    )

    print("model constructed.")

    print("loading model's state dict...")
    model.load_state_dict(torch.load(checkpoint))
    print("model's state dict loaded.")

    split = ""
    if "train" in data:
        split = "train"
    elif "dev" in data or "val" in data:
        split = "dev"
    elif "test" in data:
        split = "test"

    # Grab all sentences from a file, tokenize and align labels with (subword) tokens
    model = model.cuda()
    mentions = {}
    print("extracting mentions...")
    for i, file in enumerate(list(dataset.data.keys())):
        print('Extracting mentions from file {} ({})'.format(file, i))
        text = dataset.data[file]
        mentions[file] = {}
        mentions[file]['tokenized_text'] = text
        test_dataset = batch_and_tokenize_data(tokenizer, text, batch_size)
        preds = []
        for batch in test_dataset:
            preds += tag(model, batch, label_vocab)
        sent_idx = 0
        for sent, sent_pred in zip(text, preds):
            res = extract_entities(sent_pred, task)
            sent_entities = [{"mention": sent[start_:(end_ + 1)],
                              "start_offset": start_, "end_offset": end_, "pred_type": NE}
                             for (start_, end_), NE in res.items()]

            mentions[file][sent_idx] = sent_entities
            sent_idx += 1

        if i % 20000 == 0 and i != 0:
            outpath = os.path.join(out_dir, '{}_{}_mentions_{}.pkl'.format(outcome, split, i))
            pickle.dump(mentions, open(outpath, 'wb'))
            mentions = {}
    if mentions:
        if i < 20000:
            outpath = os.path.join(out_dir, '{}_{}_mentions.pkl'.format(outcome, split))
        else:
            outpath = os.path.join(out_dir, '{}_{}_mentions_{}.pkl'.format(outcome, split, i))
        pickle.dump(mentions, open(outpath, 'wb'))
    print("mentions extracting done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, action='store', required=True,
                        help='Provide path to csv file containing raw texts')
    parser.add_argument('--out_dir', type=str, action='store', required=True,
                        help='Provide path to directory to store outputs')
    parser.add_argument('--model_name_or_path', type=str, action='store', required=True,
                        help='Path to pretrained LM to be used')
    parser.add_argument('--checkpoint', type=str, action='store', required=True,
                        help='Path to checkpoint to load model weights from')
    parser.add_argument('--task', type=str, action='store', required=True,
                        help='Choose whether to do PICO or i2b2 concept tagging')
    parser.add_argument('--batch_size', type=int, action='store', default=16, help='Specify batch size')
    parser.add_argument('--outcome', type=str, action='store', required=True, help='name of the outcome')
    args = parser.parse_args()

    run_tagger(**vars(args))
