'''
Code for NER and PICO tagging. Model: Pretrained LM + linear layer
Run: python pico_trainer.py --data_dir <DATA DIR> --out_dir <OUTPUT DIR> --model_name_or_path <LM NAME> --task <pico/i2b2> --do_train --do_test
'''

import os
import argparse
import random
import numpy as np
from collections import Counter

import torch
import torch.optim as optim
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)
from data_loader import PICODataset, i2b2Dataset


def compute_macro_f1(predictions, references, label_list):
    results = {}
    total_f1 = 0.0
    for label in label_list:
        gold_count = 0
        pred_count = 0
        correct_count = 0
        for pred_list, ref_list in zip(predictions, references):
            for pred, ref in zip(pred_list, ref_list):
                if ref == label:
                    gold_count += 1
                if pred == label:
                    pred_count += 1
                    if pred == ref:
                        correct_count += 1
        label_prec = correct_count / float(pred_count) if pred_count != 0 else 0.0
        label_rec = correct_count / float(gold_count) if gold_count != 0 else 0.0
        label_f1 = (2 * label_prec * label_rec) / (label_prec + label_rec) if label_prec != 0 and label_rec != 0 else 0.0
        results[label] = label_f1
        total_f1 += label_f1
    total_f1 /= len(label_list)
    results['overall_f1'] = total_f1
    return results


def extract_entities(sequence):
    entities = {}
    starts = [i for i,x in enumerate(sequence) if x.startswith('B')]
    ends = []
    for idx in starts:
        if idx == len(sequence)-1 or not sequence[idx+1].startswith('I'):
            ends.append(idx)
            continue
        cur_idx = idx + 1
        while cur_idx < len(sequence) and sequence[cur_idx].startswith('I'):
            cur_idx += 1
        ends.append(cur_idx-1)
    if len(starts) != len(ends):
        print('Missing end indices for some predictions!!')
    offsets = list(zip(starts, ends))
    for offset in offsets:
        tag = sequence[offset[0]].split('-')[1]
        entities[offset] = tag
    return entities


def compute_exact_f1(predictions, references, label_list):
    results = {}
    pred_count = 0.0
    ref_count = 0.0
    correct_count = 0.0
    num_tags = (len(label_list)-1)/2
    per_tag_pred = Counter()
    per_tag_ref = Counter()
    per_tag_correct = Counter()
    for pred_list, ref_list in zip(predictions, references):
        pred_entities = extract_entities(pred_list)
        ref_entities = extract_entities(ref_list)
        pred_count += len(pred_entities)
        ref_count += len(ref_entities)
        per_tag_pred.update(list(pred_entities.values()))
        per_tag_ref.update(list(ref_entities.values()))
        matched_spans = set(pred_entities.keys()).intersection(set(ref_entities.keys()))  # Find entities that match boundaries exactly
        for span in matched_spans:
            if pred_entities[span] == ref_entities[span]:  # Check that type also matches
                correct_count += 1
                per_tag_correct.update([pred_entities[span]])
    rec = correct_count / ref_count if ref_count != 0 else 0.0
    prec = correct_count / pred_count if pred_count != 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if prec != 0 and rec != 0 else 0.0
    for label in per_tag_ref:
        tag_prec = per_tag_correct[label] / float(per_tag_pred[label]) if per_tag_pred[label] != 0 else 0.0
        tag_rec = per_tag_correct[label] / float(per_tag_ref[label]) if per_tag_ref[label] != 0 else 0.0
        tag_f1 = (2 * tag_prec * tag_rec) / (tag_prec + tag_rec) if tag_rec != 0 and tag_prec != 0 else 0.0
        results[label] = tag_f1
    results['overall_f1'] = f1
    return results


def train(model, train_data, dev_data, out_dir, label_vocab, epochs, lr, task='pico'):
    label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    print(label_list)
    step = 0
    prev_dev_loss = 10000
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_data:
            optimizer.zero_grad()
            batch = {x:y.cuda() for x,y in batch.items()}   # Tensors need to be moved to GPU
            outputs = model(**batch)   # Batch is a dictionary from HF that needs to be unpacked
            loss = outputs[0]
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1
            if step%1000 == 0:
                print('Completed {} training steps'.format(step))
        epoch_loss /= len(train_data)
        print('Training loss after epoch {}: {}'.format(epoch, epoch_loss))
        # Checkpoint model after each epoch anyway, in addition to storing best loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, os.path.join(out_dir, 'checkpoints/checkpoint_{}.pt'.format(epoch)))
        dev_loss = test(model, dev_data, label_vocab, epoch=epoch, return_loss=True, task=task)
        if dev_loss < prev_dev_loss:
            prev_dev_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
        scheduler.step(dev_loss)


def test(model, dev_data, label_vocab, epoch=0, return_loss=False, task='pico'):
    model.eval()
    label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
    dev_loss = 0.0
    model_predictions = []
    gold_labels = []
    for batch in dev_data:
        batch = {x:y.cuda() for x,y in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        dev_loss += loss.item()
        cur_preds = np.argmax(outputs[1].detach().cpu().numpy(), axis=2)
        labels = batch['labels'].cpu().numpy()
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(cur_preds, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(cur_preds, labels)
        ]
        model_predictions += true_predictions
        gold_labels += true_labels
    dev_loss /= len(dev_data)
    print('Validation loss after epoch {}: {}'.format(epoch, dev_loss))
    results = {}
    if task == 'pico':
        results = compute_macro_f1(model_predictions, gold_labels, label_list)
    elif task == 'i2b2':
        results = compute_exact_f1(model_predictions, gold_labels, label_list)
    print('------------------F1 Scores for Epoch {}-------------------'.format(epoch))
    print('Overall Macro F1 Score: {}'.format(results['overall_f1']))
    for label in results:
        if 'overall' in label:
            continue
        print('F1 Score for {}: {}'.format(label, results[label]))
    if return_loss:
        return dev_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, action='store', required=True, help='Provide path to data directory')
    parser.add_argument('--out_dir', type=str, action='store', required=True, help='Provide path to directory to store outputs')
    parser.add_argument('--model_name_or_path', type=str, action='store', required=True, help='Path to pretrained LM to be used')
    parser.add_argument('--task', type=str, action='store', required=True, help='Choose whether to do PICO or i2b2 concept tagging')
    parser.add_argument('--do_train', action='store_true', default=False, help='Specify if training should be performed')
    parser.add_argument('--do_test', action='store_true', default=False, help='Specify if evaluation on test data should be performed')
    parser.add_argument('--batch_size', type=int, action='store', default=16, help='Specify batch size')
    parser.add_argument('--lr', type=float, action='store', default=2e-5, help='Specify learning rate')
    parser.add_argument('--epochs', type=int, action='store', default=20, help='Specify number of epochs')
    parser.add_argument('--seed', type=int, action='store', default=42, help='Specify random seed')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    checkpoint_dir = os.path.join(args.out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    set_seed(args.seed)

    dataset = []
    if args.task == 'pico':
        dataset = PICODataset(args.data_dir)
    elif args.task == 'i2b2':
        dataset = i2b2Dataset(args.data_dir)
    else:
        print('Invalid task name provided - choose from pico or i2b2!')
    label_vocab = dataset.label_vocab

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(list(label_vocab.keys())),
        label2id=label_vocab,
        id2label={i: l for l, i in label_vocab.items()},
        finetuning_task='ner',
        cache_dir='../../cache/',
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

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        example_texts = [x['tokens'] for x in examples]
        tokenized_inputs = tokenizer(
            example_texts,
            padding='max_length',
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            max_length=128,
            return_tensors='pt'
        )
        labels = []
        for i, example in enumerate(examples):
            label_seq = example['gold_seq']
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_seq[word_idx])
                # For the other tokens in a word, we set the label to -100, but we might want to change that?
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = torch.cuda.LongTensor(labels)
        return tokenized_inputs

    def batch_and_tokenize_data(examples, batch_size):
        random.shuffle(examples)
        batches = []
        for i in range(0, len(examples), batch_size):
            start = i
            end = min(start+batch_size, len(examples))
            batch = tokenize_and_align_labels(examples[start:end])
            batches.append(batch)
        return batches

    # Batch data, tokenize and align labels with (subword) tokens
    train_dataset = batch_and_tokenize_data(dataset.data["train"], args.batch_size)
    dev_dataset = batch_and_tokenize_data(dataset.data["dev"], args.batch_size)
    test_dataset = batch_and_tokenize_data(dataset.data["test"], args.batch_size)

    model = model.cuda()

    if args.do_train:
        train(model, train_dataset, dev_dataset, args.out_dir, label_vocab, args.epochs, args.lr, task=args.task)
    if args.do_test:
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pt')))
        test(model, test_dataset, label_vocab, task=args.task)
