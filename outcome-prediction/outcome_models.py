import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from transformers import BertForSequenceClassification
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention


class BertLongSelfAttention(LongformerSelfAttention):

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        is_index_masked = attention_mask < 0
        is_index_masked = is_index_masked.squeeze(1).squeeze(1)
        attention_mask = attention_mask.squeeze(1).squeeze(1)
        # print('Running self-attention layer #: {}'.format(self.layer_id))
        return super().forward(hidden_states, \
                attention_mask=attention_mask, \
                is_index_masked=is_index_masked) # output_attentions=output_attentions [Arg not present in v4.1.1]


class BertLongForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)


class LitAugPredictorCrossenc(nn.Module):

    def __init__(self, bert_config, bert_model, topk, strategy='average'):
        super().__init__()
        self.bert_model = bert_model
        self.bert_config = bert_config
        self.topk = topk
        self.strategy = strategy
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pubmed_docs=None,
        pubmed_doc_weights=None
    ):
        note_lit_reps = []
        if 'vote' in self.strategy:
            prob_matrices = []
            for doc_batch in pubmed_docs:
                doc_batch = {x:y.cuda() for x,y in doc_batch.items()}
                cur_logits = self.bert_model(**doc_batch)[0]
                cur_logits_softmax = self.softmax(cur_logits)
                prob_matrices.append(cur_logits_softmax)
            averaged_probs = None
            if self.strategy == 'softvote':
                averaged_probs = torch.mean(torch.stack(prob_matrices), dim=0)
            if self.strategy == 'weightvote':
                if len(prob_matrices) == 1:
                    averaged_probs = torch.mean(torch.stack(prob_matrices), dim=0)
                else:
                    weighted_matrices = []
                    total_weight = torch.zeros(prob_matrices[0].size()).cuda()
                    for prob_matrix, weights in zip(prob_matrices, pubmed_doc_weights):
                        weights = torch.cuda.FloatTensor(weights).unsqueeze(1).repeat(1, self.bert_config.num_labels)
                        weighted_matrices.append(weights * prob_matrix)
                        total_weight += weights
                    weighted_matrices = [x/total_weight for x in weighted_matrices]
                    averaged_probs = torch.sum(torch.stack(weighted_matrices), dim=0)
            averaged_log_probs = torch.log(averaged_probs)
            return (None, averaged_log_probs)
        if self.strategy == 'average':
            rep_list = []
            for doc_batch in pubmed_docs:
                doc_batch = {x:y.cuda() for x,y in doc_batch.items()}
                cur_outputs = self.bert_model.bert(**doc_batch)[1]   # 0 - last state, 1 - pooled output
                rep_list.append(cur_outputs)
            final_lit_rep = torch.mean(torch.stack(rep_list), dim=0)
            logits = self.bert_model.classifier(final_lit_rep)
            return (None, logits)
        if self.strategy == 'weightaverage':
            rep_list = []
            total_weight = torch.zeros((input_ids.size()[0], self.bert_config.hidden_size)).cuda()
            for doc_batch, weights in zip(pubmed_docs, pubmed_doc_weights):
                doc_batch = {x:y.cuda() for x,y in doc_batch.items()}
                cur_outputs = self.bert_model.bert(**doc_batch)[1]
                weights = torch.cuda.FloatTensor(weights).unsqueeze(1).repeat(1, self.bert_config.hidden_size)
                rep_list.append(weights * cur_outputs)
                total_weight += weights
            rep_list = [x/total_weight for x in rep_list]
            averaged_reps = torch.sum(torch.stack(rep_list), dim=0)
            logits = self.bert_model.classifier(averaged_reps)
            return (None, logits)


class LitAugPredictorBienc(nn.Module):

    def __init__(self, bert_config, bert_model, topk, strategy='average'):
        super().__init__()
        self.input_size = 2 * bert_config.hidden_size  # embeddings of note + literature
        self.output_size = bert_config.num_labels
        self.bert_model = bert_model
        self.bert_config = bert_config
        self.topk = topk
        self.strategy = strategy
        self.predictor = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pubmed_docs=None,
        pubmed_doc_weights=None,
        split='train'
    ):
        note_outputs = self.bert_model.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        note_reps = note_outputs[1]
        lit_reps = []
        if len(pubmed_docs) >= 50:
            pubmed_docs = pubmed_docs[:50]
        # print(len(pubmed_docs))
        if len(pubmed_docs) == 0:
            lit_reps.append(torch.zeros(note_reps.size()).cuda())
        for doc_batch in pubmed_docs:
            doc_batch = {x:y.cuda() for x,y in doc_batch.items()}
            cur_outputs = self.bert_model.bert(**doc_batch)
            lit_reps.append(cur_outputs[1])
        if self.strategy == 'average':
            final_lit_rep = torch.mean(torch.stack(lit_reps), dim=0)
            final_rep = torch.cat([note_reps, final_lit_rep], dim=1)
            logits = self.predictor(final_rep)
            return (None, logits)
        if self.strategy == 'weightaverage':
            total_lit_rep = torch.zeros(lit_reps[0].size()).cuda()
            total_weight = torch.zeros((input_ids.size()[0], self.bert_config.hidden_size)).cuda()
            for cur_lit_rep, weights in zip(lit_reps, pubmed_doc_weights):
                weights = torch.cuda.FloatTensor(weights).unsqueeze(1).repeat(1, self.bert_config.hidden_size)
                total_weight += weights
                total_lit_rep += (weights * cur_lit_rep)
            if torch.sum(total_weight).item() != 0.0:
                total_lit_rep /= total_weight
            final_rep = torch.cat([note_reps, total_lit_rep], dim=1)
            logits = self.predictor(final_rep)
            return (None, logits)
        if self.strategy == 'softvote' or self.strategy == 'weightvote':
            prob_matrices = []
            for cur_lit_rep in lit_reps:
                cur_final_rep = torch.cat([note_reps, cur_lit_rep], dim=1)
                cur_logits = self.predictor(cur_final_rep)
                cur_logits_softmax = self.softmax(cur_logits)
                prob_matrices.append(cur_logits_softmax)
            averaged_probs = None
            if self.strategy == 'softvote':
                averaged_probs = torch.mean(torch.stack(prob_matrices), dim=0)
            if self.strategy == 'weightvote':
                if len(prob_matrices) == 1:
                    averaged_probs = torch.mean(torch.stack(prob_matrices), dim=0)
                else:
                    weighted_matrices = []
                    total_weight = torch.zeros(prob_matrices[0].size()).cuda()
                    for prob_matrix, weights in zip(prob_matrices, pubmed_doc_weights):
                        weights = torch.cuda.FloatTensor(weights).unsqueeze(1).repeat(1, self.output_size)
                        weighted_matrices.append(weights * prob_matrix)
                        total_weight += weights
                    weighted_matrices = [x/total_weight for x in weighted_matrices if torch.sum(total_weight).item() != 0.0]
                    averaged_probs = torch.sum(torch.stack(weighted_matrices), dim=0)
            averaged_log_probs = torch.log(averaged_probs)
            return (None, averaged_log_probs)


class L2RLitAugPredictorBienc(nn.Module):

    def __init__(self, bert_config, bert_model, tokenizer, topk, strategy='average', rerank_model=None, query_proj=None):
        super().__init__()
        self.input_size = 2 * bert_config.hidden_size  # embeddings of note + literature
        self.output_size = bert_config.num_labels
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.bert_config = bert_config
        self.topk = topk
        self.strategy = strategy
        self.predictor = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.cosine = nn.CosineSimilarity(dim=2)
        if rerank_model is not None:
            self.rerank_model = rerank_model
        if query_proj is not None:
            self.query_proj = query_proj
            if query_proj == 'linear':
                self.query_proj_layer = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
            if query_proj == 'transformer':
                encoder_layer = nn.TransformerEncoderLayer(d_model=bert_config.hidden_size, nhead=8)
                self.query_proj_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pubmed_docs=None,
        pubmed_doc_weights=None,
        pubmed_doc_embeds=None,
        ehr_rerank_tokens=None,
        pubmed_doc_ids=None,
        pubmed_doc_labels=None,
        split='train'
    ):
        note_question_outputs, note_question_hidden_states = None, None
        retrieval_loss = 0.0
        if hasattr(self, 'rerank_model'):
            note_question_outputs = self.rerank_model(**ehr_rerank_tokens)
            note_question_outputs = note_question_outputs['last_hidden_state'][:,0,:]
        else:
            note_question_outputs = self.bert_model.bert(**ehr_rerank_tokens)
            note_question_hidden_states = note_question_outputs[0]
            note_question_outputs = note_question_outputs[1]
        if hasattr(self, 'query_proj_layer'):
            if self.query_proj == 'linear':
                note_question_outputs = self.query_proj_layer(note_question_outputs)
            if self.query_proj == 'transformer':
                note_question_hidden_states = note_question_hidden_states.permute(1,0,2)
                note_question_outputs = self.query_proj_layer(note_question_hidden_states)
                note_question_outputs = torch.mean(note_question_outputs.permute(1,0,2), dim=1)
        if hasattr(self, 'query_loss'):
            if self.query_loss == 'pred':
                empty_lit_reps = torch.zeros(note_question_outputs.size()).cuda()
                note_question_lit_reps = torch.cat([note_question_outputs, empty_lit_reps], dim=1)
                note_question_probs = self.predictor(note_question_lit_reps)
                retrieval_loss = nn.CrossEntropyLoss()(note_question_probs, labels)
        note_question_reps = note_question_outputs.unsqueeze(1)
        note_question_rep_repeat = note_question_reps.repeat(1,pubmed_doc_embeds.size()[1],1)
        note_lit_sim = self.cosine(note_question_rep_repeat, pubmed_doc_embeds)
        # note_lit_sim = torch.nan_to_num(note_lit_sim, nan=-1.1)
        # note_lit_sim = torch.inner(note_question_rep_repeat, pubmed_doc_embeds)
        # note_lit_sim = -1 * torch.cdist(note_question_reps, pubmed_doc_embeds)
        # note_lit_sim = note_lit_sim.squeeze(1)
        corrected_note_lit_sim = torch.FloatTensor(np.nan_to_num(note_lit_sim.detach().cpu().numpy(), nan=-1.1)).cuda()
        top_doc_scores, top_doc_inds = torch.topk(corrected_note_lit_sim, self.topk, dim=1)  # Should break graph here
        if pubmed_doc_labels is not None:
            max_sim_array = torch.max(note_lit_sim.detach(), dim=1)[0].unsqueeze(-1)
            max_sim_array = max_sim_array.repeat(1,note_lit_sim.size()[-1])
            note_lit_softmax = self.softmax(note_lit_sim - max_sim_array)
            retrieval_loss -= torch.log(torch.sum(note_lit_softmax * pubmed_doc_labels))
        # Recompute note reps (without question) using outcome prediction LM
        note_outputs = self.bert_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        note_outputs = note_outputs[1]
        if hasattr(self, 'query_loss'):
            if self.query_loss == 'reg':
                retrieval_loss += nn.MSELoss()(note_question_outputs, note_outputs)
        note_reps = note_outputs.unsqueeze(1)
        if split == 'test' and torch.sum(torch.isnan(note_outputs)) > 0:
            note_reps = torch.FloatTensor(np.nan_to_num(note_reps.detach().cpu().numpy(), nan=0)).cuda()
            print('Note rep contains NaNs!!!')
        output_array = []
        for i in range(top_doc_inds.size()[0]):
            cur_doc_inds = top_doc_inds[i,:].detach().cpu().numpy().tolist()
            cur_args = (([pubmed_docs[i][0][0][x] for x in cur_doc_inds], None))
            cur_doc_input = self.tokenizer(*cur_args, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            cur_doc_input = {k:v.cuda() for k,v in cur_doc_input.items()}
            # print(cur_doc_input)
            # print(cur_doc_inds)
            # cur_doc_input = {k:torch.index_select(v.cuda(), 0, cur_doc_inds) for k,v in pubmed_docs[i].items()}
            cur_outputs = self.bert_model.bert(**cur_doc_input)[1]
            if split == 'test' and torch.sum(torch.isnan(cur_outputs)) > 0:
                cur_outputs = torch.FloatTensor(np.nan_to_num(cur_outputs.detach().cpu().numpy(), nan=0)).cuda()
            if self.strategy == 'average':
                final_lit_rep = torch.mean(cur_outputs, dim=0).unsqueeze(0)
                final_rep = torch.cat([note_reps[i,:,:], final_lit_rep], dim=1)
                logits = self.predictor(final_rep)
                max_val = max(logits.detach().cpu().numpy().tolist()[0])
                output_array.append(logits - max_val)
            if self.strategy == 'weightaverage':
                weights = top_doc_scores[i,:].unsqueeze(1).detach()
                total_weight = torch.sum(weights).item()
                final_lit_rep = []
                if split == 'test' and torch.sum(torch.isnan(cur_outputs)) > 0:
                    print('Lit rep contains NaNs!!!!')
                if math.isnan(total_weight):
                    final_lit_rep = torch.mean(cur_outputs, dim=0).unsqueeze(0)
                else:
                    final_lit_rep = torch.sum((cur_outputs * weights)/total_weight, dim=0).unsqueeze(0)
                final_rep = torch.cat([note_reps[i,:,:], final_lit_rep], dim=1)
                logits = self.predictor(final_rep)
                max_val = max(logits.detach().cpu().numpy().tolist()[0])
                output_array.append(logits - max_val)
            if 'vote' in self.strategy:
                cur_note_rep = note_reps[i,:,:].repeat(self.topk,1)
                final_rep = torch.cat([cur_note_rep, cur_outputs], dim=1)
                logits = self.predictor(final_rep)
                max_val = max(logits.detach().cpu().numpy().tolist()[0])
                logits_softmax = self.softmax(logits - max_val)
                if self.strategy == 'softvote':
                    output_array.append(torch.mean(logits_softmax, dim=0))
                if self.strategy == 'weightvote':
                    weights = top_doc_scores[i,:].unsqueeze(1).detach()
                    total_weight = torch.sum(weights).item()
                    if math.isnan(total_weight):
                        output_array.append(torch.mean(logits_softmax, dim=0))
                    else:
                        output_array.append(torch.sum((logits_softmax * weights)/total_weight, dim=0))
        final_output = torch.stack(output_array).squeeze(1)
        if 'vote' in self.strategy:
            final_output = torch.log(final_output)
        return (retrieval_loss, final_output)
