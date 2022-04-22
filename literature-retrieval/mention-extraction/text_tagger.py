'''
Code to dump NER or PICO tags for raw text. Model: Pretrained LM + linear layer
Run: python text_tagger.py --data <RAW TEXT CSV> --out_dir <OUTPUT DIR> --model_name_or_path <LM NAME> --checkpoint <MODEL WEIGHT FILE> --task <pico/i2b2>
'''

import os
import argparse
import random
import numpy as np
import pickle
from collections import Counter

import torch
import torch.optim as optim
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)
from data_loader import RawTextDataset

    
def test(model, dev_data, label_vocab, epoch=0, return_loss=False, task='pico'):
    model.eval()
    label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
    batch = {x:y.cuda() for x,y in dev_data.items()}
    outputs = model(**batch)
    cur_preds = np.argmax(outputs[1].detach().cpu().numpy(), axis=2)
    labels = batch['labels'].cpu().numpy()
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(cur_preds, labels)
    ]
    return true_predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, action='store', required=True, help='Provide path to csv file containing raw texts')
    parser.add_argument('--out_dir', type=str, action='store', required=True, help='Provide path to directory to store outputs')
    parser.add_argument('--model_name_or_path', type=str, action='store', required=True, help='Path to pretrained LM to be used')
    parser.add_argument('--checkpoint', type=str, action='store', required=True, help='Path to checkpoint to load model weights from')
    parser.add_argument('--task', type=str, action='store', required=True, help='Choose whether to do PICO or i2b2 concept tagging')
    parser.add_argument('--batch_size', type=int, action='store', default=16, help='Specify batch size')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dataset = RawTextDataset(args.data, args.task)
    label_vocab = dataset.label_vocab

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(list(label_vocab.keys())),
        label2id=label_vocab,
        id2label={i: l for l, i in label_vocab.items()},
        finetuning_task='ner',
        cache_dir='../../cache',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir='../../cache',
        use_fast=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir='../../cache',
    )

    model.load_state_dict(torch.load(args.checkpoint))

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples,
            padding='max_length',
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            max_length=128,
            return_tensors='pt'
        )
        labels = []
        for i, example in enumerate(examples):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set a dummy label ("O") for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(0)
                # For the other tokens in a word, we set the label to -100, but we might want to change that?
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = torch.cuda.LongTensor(labels)
        return tokenized_inputs

    def batch_and_tokenize_data(examples, batch_size):
        batches = []
        for i in range(0, len(examples), batch_size):
            start = i
            end = min(start+batch_size, len(examples))
            batch = tokenize_and_align_labels(examples[start:end])
            batches.append(batch)
        return batches

    # Grab all sentences from a file, tokenize and align labels with (subword) tokens
    model = model.cuda()
    mentions = {}
    for i, file in enumerate(list(dataset.data.keys())):
        print('Extracting mentions from file {} ({})'.format(file, i))
        text = dataset.data[file]
        mentions[file] = {}
        test_dataset = batch_and_tokenize_data(text, args.batch_size)
        preds = []
        for batch in test_dataset:
            preds += test(model, batch, label_vocab, task=args.task)
        sent_idx = 0
        for sent, sent_pred in zip(text, preds):
            mentions[file][sent_idx] = []
            starts = [i for i,x in enumerate(sent_pred) if x.startswith('B')] if args.task == 'i2b2' \
                    else [i for i,x in enumerate(sent_pred) if x.startswith('I') and not sent_pred[i-1].startswith('I')]
            ends = []
            for idx in starts:
                if idx == len(sent_pred)-1 or not sent_pred[idx+1].startswith('I'):
                    ends.append(idx)
                    continue
                cur_idx = idx + 1
                while cur_idx < len(sent_pred) and sent_pred[cur_idx].startswith('I'):
                    cur_idx += 1
                ends.append(cur_idx-1)
            if len(starts) != len(ends):
                print('Missing end indices for some predictions!!')
            offsets = list(zip(starts, ends))
            for offset in offsets:
                start, end = offset
                tag = sent_pred[offset[0]].split('-')[1]
                mentions[file][sent_idx].append({'mention': sent[start:end+1], 'start_offset':start, 'end_offset': end, 'pred_type': tag})
            sent_idx += 1
        if i % 20000 == 0 and i != 0:
            pickle.dump(mentions, open(os.path.join(args.out_dir, 'mentions_{}.pkl'.format(i)), 'wb'))
            mentions = {}
    if mentions:
        pickle.dump(mentions, open(os.path.join(args.out_dir, 'mentions_remaining.pkl'), 'wb'))
