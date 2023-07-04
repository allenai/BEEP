import torch
import numpy as np

def tokenize_and_align_labels(tokenizer, examples):
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
            # For the other tokens in a word, we set the label to -100, but we might want to change that? #TODO why, actuallu???
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = torch.cuda.LongTensor(labels)
    return tokenized_inputs

def batch_and_tokenize_data(tokenizer, examples, batch_size):
    batches = []
    for i in range(0, len(examples), batch_size):
        start = i
        end = min(start + batch_size, len(examples))
        batch = tokenize_and_align_labels(tokenizer, examples[start:end])
        batches.append(batch)
    return batches

def extract_entities(sequence, task):
    entities = {}
    starts = [i for i,x in enumerate(sequence) if x.startswith('B')]
    starts = [i for i, x in enumerate(sequence) if x.startswith('B')] if task == 'i2b2' \
        else [i for i, x in enumerate(sequence) if x.startswith('I') and not sequence[i - 1].startswith('I')]
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

def tag(model, data, label_vocab):
    with torch.no_grad():
        model.eval()
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        batch = {x:y.cuda() for x,y in data.items()}
        outputs = model(**batch)
        cur_preds = np.argmax(outputs[1].detach().cpu().numpy(), axis=2)
        labels = batch['labels'].cpu().numpy()
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(cur_preds, labels)
        ]
    return true_predictions