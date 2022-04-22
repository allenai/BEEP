import argparse
import random
import os
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
import setproctitle

from data_loader import EHRDataset
from transformers import AdamW, BertConfig, BertTokenizer, BertForSequenceClassification, \
        AutoTokenizer, AutoConfig, AutoModel, BertTokenizerFast, set_seed, get_linear_schedule_with_warmup
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from outcome_models import BertLongForSequenceClassification, LitAugPredictorBienc, LitAugPredictorCrossenc, L2RLitAugPredictorBienc

from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from shutil import copyfile

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

def create_long_model(init_model, save_model_to, attention_window, max_pos, num_labels):
    config = BertConfig.from_pretrained(init_model, 
        num_labels=num_labels, 
        label2id={x:x for x in range(num_labels)}, 
        id2label={x:x for x in range(num_labels)}
    )
    model = BertForSequenceClassification.from_pretrained(init_model, config=config)
    tokenizer = BertTokenizerFast.from_pretrained(init_model, model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.bert.embeddings.position_embeddings.weight
        k += step
    model.bert.embeddings.position_embeddings.weight.data = new_pos_embed
    model.bert.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.bert.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)


def train(model, train_data, dev_data, out_dir, epochs, lr, class_weights, acc_steps, strategy, config_string):
    # print('Dropout default" {}'.format(model.config.hidden_dropout_prob))
    weights = torch.cuda.FloatTensor([x[1] for x in list(sorted(class_weights.items(), key=lambda x:x[0]))])
    weighted_ce_loss = nn.CrossEntropyLoss(weight=weights)
    if 'vote' in strategy:
        weighted_ce_loss = nn.NLLLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    if args.use_warmup:
        # optimizer = AdamW(model.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        print('Using linear schedule with warmup for {} steps'.format(args.warmup_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, epochs*len(train_data))
    step = 0
    prev_dev_loss = 10000
    prev_auroc = -10000
    batch_size = len(train_data[0]['ehr_id'])
    acc_factor = acc_steps/batch_size
    for epoch in range(epochs):
        random.shuffle(train_data)
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        for batch in train_data:
            gpu_batch = {x:y.cuda() for x,y in batch.items() if x not in ['ehr_id', 'pubmed_docs', 'pubmed_doc_weights', 'ehr_rerank_tokens', 'pubmed_doc_ids']}
            if 'pubmed_docs' in batch:
                gpu_batch['pubmed_docs'] = batch['pubmed_docs']
                gpu_batch['pubmed_doc_weights'] = batch['pubmed_doc_weights']
            if 'pubmed_doc_ids' in batch:
                gpu_batch['pubmed_doc_ids'] = batch['pubmed_doc_ids']
            if 'ehr_rerank_tokens' in batch:
                gpu_batch['ehr_rerank_tokens'] = {x:y.cuda() for x,y in batch['ehr_rerank_tokens'].items()}
            outputs = model(**gpu_batch)
            logits = outputs[1]
            wloss = weighted_ce_loss(logits, gpu_batch["labels"])
            if outputs[0] is not None:
                wloss += outputs[0]
            wloss /= acc_factor
            epoch_loss += wloss.item()
            wloss.backward()
            step += batch_size
            if step%acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step%800 == 0:
                print('Completed {} training steps'.format(step))
                dev_loss, auroc = test(model, dev_data, epoch=epoch, return_loss=True, class_weights=class_weights, strategy=strategy)
                tune.report(dev_loss=dev_loss)
                tune.report(auroc=auroc)
                tune.report(step=step)
                if not args.stop_on_roc and dev_loss < prev_dev_loss:
                    prev_dev_loss = dev_loss
                    if os.path.exists(os.path.join(out_dir, 'checkpoint_1_{}.pt'.format(config_string))):
                        copyfile(os.path.join(out_dir, 'checkpoint_1_{}.pt'.format(config_string)),
                                os.path.join(out_dir, 'checkpoint_2_{}.pt'.format(config_string)))
                        copyfile(os.path.join(out_dir, 'best_model_{}.pt'.format(config_string)),
                                os.path.join(out_dir, 'checkpoint_1_{}.pt'.format(config_string)))
                        torch.save(model.state_dict(), os.path.join(out_dir, 'best_model_{}.pt'.format(config_string)))
                    elif os.path.exists(os.path.join(out_dir, 'best_model_{}.pt'.format(config_string))):
                        print('Need to move best model')
                        copyfile(os.path.join(out_dir, 'best_model_{}.pt'.format(config_string)),
                                os.path.join(out_dir, 'checkpoint_1_{}.pt'.format(config_string)))
                        torch.save(model.state_dict(), os.path.join(out_dir, 'best_model_{}.pt'.format(config_string)))
                    else:
                        torch.save(model.state_dict(), os.path.join(out_dir, 'best_model_{}.pt'.format(config_string)))
                if args.stop_on_roc and auroc > prev_auroc:
                    prev_auroc = auroc
                    torch.save(model.state_dict(), os.path.join(out_dir, 'best_model_{}.pt'.format(config_string)))
                if not args.use_warmup:
                    if not args.stop_on_roc:
                        scheduler.step(dev_loss)
                    else:
                        scheduler.step(auroc)
                else:
                    print('Different step for linear warmup')
                    scheduler.step()
        epoch_loss /= (len(train_data)/acc_factor)
        print('Training loss after epoch {}: {}'.format(epoch, epoch_loss))
        # NOTE: Uncomment this to enable per-epoch checkpointing
        # However, this is likely to occupy a lot of space during hyperparameter sweeps
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': epoch_loss,
        # }, os.path.join(out_dir, 'checkpoints/checkpoint_{}_{}.pt'.format(epoch, config_string)))
        dev_loss, auroc = test(model, dev_data, epoch=epoch, return_loss=True, class_weights=class_weights, strategy=strategy)
        tune.report(dev_loss=dev_loss)
        tune.report(auroc=auroc)
        tune.report(step=step)
        if dev_loss < prev_dev_loss:
            prev_dev_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model_{}.pt'.format(config_string)))
        scheduler.step(dev_loss)


def test(model, dev_data, epoch=0, return_loss=False, class_weights=None, strategy='average'):
    model.eval()
    unique_labels = list(class_weights.keys())
    weights = torch.cuda.FloatTensor([x[1] for x in list(sorted(class_weights.items(), key=lambda x:x[0]))])
    weighted_ce_loss = nn.CrossEntropyLoss(weight=weights)
    if 'vote' in strategy:
        weighted_ce_loss = nn.NLLLoss(weight=weights)
    softmax = nn.Softmax(dim=1)
    dev_loss = 0.0
    all_preds = []
    all_pred_probs = []
    all_labels = []
    all_ids = []
    all_pred_probs_dump = []
    for batch in dev_data:
        gpu_batch = {x:y.cuda() for x,y in batch.items() if x not in ['ehr_id', 'pubmed_docs', 'pubmed_doc_weights', 'ehr_rerank_tokens', 'pubmed_doc_ids']}
        if 'pubmed_docs' in batch:
            gpu_batch['pubmed_docs'] = batch['pubmed_docs']
            gpu_batch['pubmed_doc_weights'] = batch['pubmed_doc_weights']
        if 'pubmed_doc_ids' in batch:
            gpu_batch['pubmed_doc_ids'] = batch['pubmed_doc_ids']
        if 'ehr_rerank_tokens' in batch:
            gpu_batch['ehr_rerank_tokens'] = {x:y.cuda() for x,y in batch['ehr_rerank_tokens'].items()}
        outputs = model(**gpu_batch)
        logits = outputs[1]
        all_preds += torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        probs = softmax(logits) if 'average' in strategy else torch.exp(logits)
        all_pred_probs_dump += probs.detach().cpu().numpy().tolist()
        probs = probs if len(unique_labels) > 2 else probs[:,1]
        wloss = weighted_ce_loss(logits, gpu_batch["labels"])
        dev_loss += wloss.item()
        all_pred_probs += probs.detach().cpu().numpy().tolist()
        all_labels += gpu_batch["labels"].cpu().numpy().tolist()
        all_ids += batch['ehr_id']
    prediction_dict = dict(zip(all_ids, all_preds))
    pred_prob_dict = dict(zip(all_ids, all_pred_probs_dump))
    if not return_loss and args.dump_test_preds:   # return_loss flag is not used for the test data
        pickle.dump(prediction_dict, open(os.path.join(args.out_dir, 'dev_predictions.pkl'), 'wb'))
        pickle.dump(pred_prob_dict, open(os.path.join(args.out_dir, 'dev_probabilities.pkl'), 'wb'))
    auroc, f1, mf1 = compute_classification_metrics(all_preds, all_pred_probs, all_labels)
    dev_loss /= len(dev_data)
    print('Validation loss after epoch {}: {}'.format(epoch, dev_loss))
    print('------------------Validation Scores for Epoch {}-------------------'.format(epoch))
    print('AUROC: {}'.format(auroc))
    print('Micro F1: {}'.format(f1))
    print('Macro F1: {}'.format(mf1))
    if return_loss:
        return dev_loss, auroc


def compute_classification_metrics(preds, probs, labels):
    unique_labels = set(labels)
    probs = np.array(probs)
    labels = np.array(labels)
    preds = np.array(preds)
    roc_auc = roc_auc_score(y_true=labels, y_score=probs, average="macro", multi_class="ovo")
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    mf1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return roc_auc, f1, mf1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, action='store', required=True, help='Path to training file')
    parser.add_argument('--dev', type=str, action='store', required=True, help='Path to development file')
    parser.add_argument('--test', type=str, action='store', required=True, help='Path to test file')
    parser.add_argument('--lit_dir', type=str, action='store', help='Path to directory containing literature ')
    parser.add_argument('--init_model', type=str, action='store', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', \
            help='Pretrained model to initialize weights from')
    parser.add_argument('--rerank_model', type=str, action='store', help='Pretrained model to initialize reranker weights from')
    parser.add_argument('--rerank_checkpoint', type=str, action='store', help='Checkpoint to load reranker weights from')
    parser.add_argument('--longmodel_dir', type=str, action='store', help='Path to dump longformer version of model')
    parser.add_argument('--out_dir', type=str, action='store', required=True, help='Provide path to directory to store outputs')
    parser.add_argument('--do_train', action='store_true', default=False, help='Specify if training should be performed')
    parser.add_argument('--do_test', action='store_true', default=False, help='Specify if evaluation on test data should be performed')
    parser.add_argument('--checkpoint', type=str, action='store', help='Path to checkpoint to load model weights from')
    parser.add_argument('--attention_window', type=int, action='store', default=512, help='Attention window size')
    parser.add_argument('--max_pos', type=int, action='store', default=4096, help='Maximum position embedding size')
    parser.add_argument('--batch_size', type=int, action='store', default=1, help='Specify batch size')
    parser.add_argument('--lr', type=float, action='store', default=2e-5, help='Specify learning rate')
    parser.add_argument('--epochs', type=int, action='store', default=20, help='Specify number of epochs')
    parser.add_argument('--seed', type=int, action='store', default=42, help='Specify random seed')
    parser.add_argument('--accumulation_steps', type=int, action='store', default=32, help='Specify number of steps for gradient accumulation')
    parser.add_argument('--num_top_docs', type=float, action='store', default=1, help='Number of top ranked abstracts from PubMed to include')
    parser.add_argument('--strategy', type=str, action='store', default='average', help='Strategy to use to combine literature with EHR')
    parser.add_argument('--enc_strategy', type=str, action='store', default='bienc', help='Encoding strategy to use for notes and articles (bienc/crossenc)')
    parser.add_argument('--use_warmup', action='store_true', default=False, help='Choose whether to use LR warmup or not')
    parser.add_argument('--warmup_steps', type=int, action='store', default=5000, help='Choose number of warmup steps')
    parser.add_argument('--stop_on_roc', action='store_true', default=False, help='Use AUROC as early stopping metric')
    parser.add_argument('--dump_test_preds', action='store_true', default=False, help='Dump predictions on test set')
    parser.add_argument('--use_pico', action='store_true', default=False, help='Add PICO highlights to chosen literature docs')
    parser.add_argument('--doc_embeds', type=str, action='store', help='Embeddings of top ranked abstracts for learning to retrieve')
    parser.add_argument('--l2r_top_docs', type=int, action='store', default=5, help='Number of top documents to chosse in learning to retrieve')
    parser.add_argument('--outcome', type=str, action='store', required=True, help='Choose outcome to predict (pmv/los/mortality)')
    parser.add_argument('--retrieval_labels', type=str, action='store', help='Path to file containing pseudo labels for retrieval training L2R')
    parser.add_argument('--query_proj', type=str, action='store', help='Projection layer to use for queries in L2R')
    parser.add_argument('--query_loss', type=str, action='store', help='Direct loss term for query encoding (pred/reg)')
    args = parser.parse_args()

    if args.longmodel_dir is not None and not os.path.exists(args.longmodel_dir):
        os.makedirs(args.longmodel_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    checkpoint_dir = os.path.join(args.out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    set_seed(args.seed)
    setproctitle.setproctitle("python")

    def preprocess_function(tokenizer, examples, split, topk, rerank_tokenizer=None):
        data_args = (([x[1] for x in examples], None))
        max_length = args.max_pos if args.longmodel_dir is not None else 512
        result = tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        result["labels"] = torch.LongTensor([(x[2]) for x in examples])
        result["ehr_id"] = [(x[0]) for x in examples]
        if args.doc_embeds is not None:
            if rerank_tokenizer is not None:
                data_args = (([outcome_questions[args.outcome] + x[1] for x in examples], None))
                result["ehr_rerank_tokens"] = rerank_tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
            else:
                data_args = (([outcome_questions[args.outcome] + x[1] for x in examples], None))
                result["ehr_rerank_tokens"] = tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        if args.lit_dir is not None and args.doc_embeds is None:
            result["pubmed_docs"] = []
            result["pubmed_doc_weights"] = []
            k_range = int(topk) if topk >= 1 else max([len(x[-1]) for x in examples])
            if k_range > 0:
                if args.enc_strategy == 'bienc':
                    for k in range(k_range):
                        result["pubmed_doc_weights"].append([x[-1][k][2] if len(x[-1]) > k else 0.0 for x in examples])
                        data_args = (([x[-1][k][1] if len(x[-1]) > k else '' for x in examples], None))
                        result["pubmed_docs"].append(tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt'))
                if args.enc_strategy == 'crossenc':
                    for k in range(k_range):
                        result["pubmed_doc_weights"].append([x[-1][k][2] if len(x[-1]) > k else 0.0 for x in examples])
                        data_args = (([x[1] for x in examples], [x[-1][k][1] if len(x[-1]) > k else '' for x in examples]))
                        result["pubmed_docs"].append(tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt'))
        if args.doc_embeds is not None:
            result["pubmed_docs"] = []
            result["pubmed_doc_weights"] = []
            result["pubmed_doc_embeds"] = []
            result["pubmed_doc_ids"] = []
            if args.retrieval_labels is not None and split=='train':
                result["pubmed_doc_labels"] = []
            for x in examples:
                result["pubmed_doc_ids"].append([y[0] for y in x[-1]])
                result["pubmed_doc_weights"].append([y[2] for y in x[-1]])
                data_args = (([y[1] for y in x[-1]], None))  # y[0] will be Pubmed ID of doc
                result["pubmed_docs"].append(tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt'))
                result["pubmed_doc_embeds"].append(np.vstack([x[3][y[0]] for y in x[-1]])[np.newaxis,:,:])
                if retrieval_labels is not None and split=='train':
                    result["pubmed_doc_labels"].append([retrieval_labels[x[0]][y[0]] for y in x[-1]])
            if retrieval_labels is not None and split=='train':
                result["pubmed_doc_labels"] = torch.LongTensor(np.vstack(result["pubmed_doc_labels"]))
            result["pubmed_doc_embeds"] = np.vstack(result["pubmed_doc_embeds"])
            result["pubmed_doc_embeds"] = torch.FloatTensor(result["pubmed_doc_embeds"])
        return result

    def batch_and_tokenize_data(tokenizer, examples, batch_size, split, topk, rerank_tokenizer=None):
        example_list = []
        for file in list(examples.keys()):
            example = examples[file]
            if args.lit_dir is None:
                example_list.append([file, example['ehr'], example['outcome']])
            elif args.doc_embeds is None:
                example_list.append([file, example['ehr'], example['outcome'], example['pubmed_docs']])
            else:
                example_list.append([file, example['ehr'], example['outcome'], example['pubmed_doc_embeds'], example['pubmed_docs']])
        batches = []
        # if args.longmodel_dir is not None and (split == 'dev' or split == 'test'):
        #    batch_size = 1
        for i in range(0, len(example_list), batch_size):
            start = i
            end = min(start+batch_size, len(example_list))
            batch = preprocess_function(tokenizer, example_list[start:end], split, topk, rerank_tokenizer)
            batches.append(batch)
            if len(batches) % 100 == 0:
                print('Created {} batches'.format(len(batches)))
        return batches

    def run_outcome_prediction_pipeline(config):
        outcome_questions = {'mortality': 'What is the hospital mortality? ', \
            'pmv': 'What is the probability of prolonged mechanical ventilation? ', \
            'los': 'What is the probable length of stay? '}

        dataset = EHRDataset(args.train, args.dev, args.test)
        dataset.compute_class_weights()
        if args.lit_dir is not None:
            dataset.add_relevant_literature(args.lit_dir, args.num_top_docs, args.use_pico)
            missing_lit = 0
            for doc in dataset.train_data:
                if 'pubmed_docs' not in dataset.train_data[doc]:
                    missing_lit += 1
            for doc in dataset.dev_data:
                if 'pubmed_docs' not in dataset.dev_data[doc]:
                    missing_lit += 1
            for doc in dataset.test_data:
                if 'pubmed_docs' not in dataset.test_data[doc]:
                    missing_lit += 1
            print('{} documents do not have PubMed abstracts'.format(missing_lit))
        if args.doc_embeds:
            dataset.add_literature_matrices(args.doc_embeds)
    
        num_labels = len(list(dataset.class_weights.keys()))
        retrieval_labels = None
        if args.retrieval_labels is not None:
            retrieval_labels = pickle.load(open(args.retrieval_labels, 'rb'))

        if args.longmodel_dir is not None:
            create_long_model(
                init_model=args.init_model, 
                save_model_to=args.longmodel_dir, 
                attention_window=args.attention_window, 
                max_pos=args.max_pos,
                num_labels=num_labels
            )
        model_path = args.longmodel_dir if args.longmodel_dir is not None else args.init_model
        model_config = BertConfig.from_pretrained(model_path,
            num_labels=num_labels,
            label2id={x:x for x in range(num_labels)},
            id2label={x:x for x in range(num_labels)}
        )
        tokenizer = BertTokenizerFast.from_pretrained(model_path) if 'Discharge' not in model_path \
                else AutoTokenizer.from_pretrained(model_path)
        model = BertLongForSequenceClassification.from_pretrained(model_path, config=model_config) if args.longmodel_dir is not None \
                else BertForSequenceClassification.from_pretrained(model_path, config=model_config)
        rerank_config, rerank_tokenizer, rerank_model = None, None, None
        if args.rerank_model is not None:
            rerank_label_vocab = {'Relevant': 1, 'Irrelevant': 0}
            rerank_config = AutoConfig.from_pretrained(
                args.rerank_model,
                num_labels=len(list(rerank_label_vocab.keys())),
                label2id=rerank_label_vocab,
                id2label={i: l for l, i in rerank_label_vocab.items()},
                cache_dir='../cache',
            ) 
            rerank_tokenizer = AutoTokenizer.from_pretrained(
                args.rerank_model,
                cache_dir='../cache',
                use_fast=True,
            )
            rerank_model = AutoModel.from_pretrained(
                args.rerank_model,
                from_tf=bool(".ckpt" in args.rerank_model),
                config=rerank_config,
                cache_dir='../cache',
            )
            special_tokens_dict = {'additional_special_tokens': ['[ENTSEP]']}
            num_added_toks = rerank_tokenizer.add_special_tokens(special_tokens_dict)
            rerank_model.resize_token_embeddings(len(rerank_tokenizer))
            if args.rerank_checkpoint is not None and args.do_train:   # Only load pretrained reranker if training is to be carried out
                rerank_model.load_state_dict(torch.load(args.rerank_checkpoint))     # Otherwise full model will contain reranker weights too
        if args.use_pico:
            special_tokens_dict = {'additional_special_tokens': ['<PAR>', '</PAR>', '<INT>', '</INT>', '<OUT>', '</OUT>']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))
            print('Added additional special tokens for PICO highlights')

        if args.lit_dir is not None and args.doc_embeds is None:
            if args.enc_strategy == 'bienc':
                model = LitAugPredictorBienc(model_config, model, args.num_top_docs, args.strategy)
            elif args.enc_strategy == 'crossenc':
                model = LitAugPredictorCrossenc(model_config, model, args.num_top_docs, args.strategy)
        if args.lit_dir is not None and args.doc_embeds is not None:
            if args.query_proj is None:
                model = L2RLitAugPredictorBienc(model_config, model, args.l2r_top_docs, args.strategy, rerank_model)
            else:
                model = L2RLitAugPredictorBienc(model_config, model, args.l2r_top_docs, args.strategy, rerank_model, args.query_proj)
            if args.query_loss is not None:
                model.query_loss = args.query_loss
        model = model.cuda()
        # print('Initialized longformer model with pretrained LM...')

        print('Started batch creation')
        train_batches = batch_and_tokenize_data(tokenizer, dataset.train_data, args.batch_size, 'train', args.num_top_docs, rerank_tokenizer)
        print('Created {} train batches'.format(len(train_batches)))
        dev_batches = batch_and_tokenize_data(tokenizer, dataset.dev_data, args.batch_size, 'dev', args.num_top_docs, rerank_tokenizer)
        print('Created {} dev batches'.format(len(dev_batches)))
        test_batches = batch_and_tokenize_data(tokenizer, dataset.test_data, args.batch_size, 'test', args.num_top_docs, rerank_tokenizer)
        print('Created {} test batches'.format(len(test_batches)))

        config_string = '{}_{}'.format(config["lr"], config["acc"])
        if args.do_train:
            train(model, train_batches, dev_batches, args.out_dir, args.epochs, config["lr"], dataset.class_weights, \
                    config["acc"], args.strategy, config_string)
        if args.do_test:
            if args.checkpoint is not None:
                if 'checkpoint' in args.checkpoint:
                    full_checkpoint = torch.load(args.checkpoint)
                    model.load_state_dict(full_checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(torch.load(args.checkpoint))
            else:
                model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model_{}.pt'.format(config_string))))
            test(model, dev_batches, class_weights=dataset.class_weights, strategy=args.strategy)

    scheduler = MedianStoppingRule(time_attr='step', metric='dev_loss', mode='min', grace_period=10000, min_time_slice=800)
    analysis = tune.run(
        run_outcome_prediction_pipeline, 
        num_samples=1, 
        config={
            "num_workers": 2, 
            "lr": tune.grid_search([5e-4, 1e-5, 5e-5, 1e-6, 5e-6]), 
            "acc": tune.grid_search([10, 20])
        },
        resources_per_trial={'gpu': 1},
        scheduler=scheduler
    )
    print("best config: ", analysis.get_best_config(metric="dev_loss", mode="min"))
    best_config = analysis.get_best_config(metric="dev_loss", mode="min")
    config_string = '{}_{}'.format(best_config["lr"], best_config["acc"])
    model_path = args.longmodel_dir if args.longmodel_dir is not None else args.init_model
    num_labels = 2 if args.outcome != 'los' else 4
    model_config = BertConfig.from_pretrained(model_path,
            num_labels=num_labels,
            label2id={x:x for x in range(num_labels)},
            id2label={x:x for x in range(num_labels)}
        )
    tokenizer = BertTokenizerFast.from_pretrained(model_path) if 'Discharge' not in model_path \
            else AutoTokenizer.from_pretrained(model_path)
    model = BertLongForSequenceClassification.from_pretrained(model_path, config=model_config) if args.longmodel_dir is not None \
            else BertForSequenceClassification.from_pretrained(model_path, config=model_config)
    dataset = EHRDataset(args.train, args.dev, args.test)
    dataset.compute_class_weights()
    if args.lit_dir is not None:
        dataset.add_relevant_literature(args.lit_dir, best_config["k"], args.use_pico)
        if args.doc_embeds:
            dataset.add_literature_matrices(args.doc_embeds)
    test_batches = batch_and_tokenize_data(tokenizer, dataset.test_data, args.batch_size, 'test', args.num_top_docs, None)
    if args.lit_dir is not None and args.doc_embeds is None:
        if args.enc_strategy == 'bienc':
            model = LitAugPredictorBienc(model_config, model, args.num_top_docs, args.strategy)
        elif args.enc_strategy == 'crossenc':
            model = LitAugPredictorCrossenc(model_config, model, args.num_top_docs, args.strategy)
    if args.lit_dir is not None and args.doc_embeds is not None:
        if args.query_proj is None:
            model = L2RLitAugPredictorBienc(model_config, model, args.l2r_top_docs, args.strategy, rerank_model)
        else:
            model = L2RLitAugPredictorBienc(model_config, model, args.l2r_top_docs, args.strategy, rerank_model, args.query_proj)
        if args.query_loss is not None:
            model.query_loss = args.query_loss
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model_{}.pt'.format(config_string))))
    test(model, test_batches, class_weights=dataset.class_weights, strategy=args.strategy)
