import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification
from sklearn.model_selection import KFold
from transformers import BertPreTrainedModel, RobertaModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torchcrf import CRF

from data_processing import TorchData, train_and_tags

cat_map = {'Unbalanced_power_relations':0, 'Shallow_solution':1, 
               'Presupposition':2, 'Authority_voice':3, 'Metaphors':4,
               'Compassion':5, 'The_poorer_the_merrier':6}

tag2id = {}


#https://github.com/shushanxingzhe/transformers_ner
class RobertaCRF(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

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
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            log_likelihood, tags = self.crf(logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags


def compute_metrics_token(pred):
    preds, labels = pred
    preds = np.argmax(preds, axis=2)

    b = tag2id['B']
    i = tag2id['I']
    o = tag2id['O']

    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for j in range(len(labels)):
        if (b in labels[j] and b in preds[j]) or (b in labels[j] and i in preds[j]):
            tp += 1
        elif (b in labels[j] and b not in preds[j]) or (b in labels[j] and i not in preds[j]):
            fn += 1
        elif (b not in labels[j] and b in preds[j]) or (b not in labels[j] and i in preds[j]):
            fp += 1
        elif (b not in labels[j] and b not in preds[j]) or (b not in labels[j] and i not in preds[j]):
            tn += 1
    if tp+fp != 0 and tp+fn != 0:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        if precision+recall != 0:
            f1 = 2*(precision*recall)/(precision+recall)
        else:
            f1 = 0
    else:
        precision = 0
        recall = 0
        f1 = 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


#https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/pytorch/custom_datasets.ipynb
#https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb
'''def tokenize_and_align_labels(tokenizer, X, tags, keep_word_split=True):
    X = [x.split(' ') for x in X]
    tokenized_inputs = tokenizer(X, truncation=True, is_split_into_words=True)
    unique_tags = set(tag for doc in tags for tag in doc)
    global tag2id
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    tags = [[tag2id[tag] for tag in doc] for doc in tags]

    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if keep_word_split else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    return tokenized_inputs, labels'''
def tokenize(tokenizer, X, tags):
    X = [x.split(' ') for x in X]
    result = {
        'label_ids': [],
        'input_ids': [],
        'token_type_ids': [],
    }
    max_length = tokenizer.max_model_input_sizes['bert-base-cased']

    unique_tags = set(tag for doc in tags for tag in doc)
    global tag2id
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    tags = [[tag2id[tag] for tag in doc] for doc in tags]

    for tokens, label in zip(X, tags):
        tokenids = tokenizer(tokens, add_special_tokens=False, truncation=True, is_split_into_words=True)

        token_ids = []
        label_ids = []
        for ids, lab in zip(tags, label):
            if len(ids) > 1 and lab % 2 == 1:
                token_ids.extend(ids)
                chunk = [lab + 1] * len(ids)
                chunk[0] = lab
                label_ids.extend(chunk)
            else:
                token_ids.extend(ids)
                chunk = [lab] * len(ids)
                label_ids.extend(chunk)

        token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids)
        token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
        label_ids.insert(0, 0)
        label_ids.append(0)
        result['input_ids'].append(token_ids)
        result['label_ids'].append(label_ids)
        result['token_type_ids'].append(token_type_ids)

    result = tokenizer.pad(result, padding='longest', max_length=max_length, return_attention_mask=True)
    for i in range(len(result['input_ids'])):
        diff = len(result['input_ids'][i]) - len(result['label_ids'][i])
        result['label_ids'][i] += [0] * diff
    return result['input_ids'], result['label_ids']


def main():
    pcl = open('data/dontpatronizeme_categories.tsv').read()
    Xs, tags = train_and_tags(pcl)

    precision_all = []
    recall_all = []
    f1_all = []
    for i in range(len(tags)):
        X = np.array(Xs[i])
        y = np.array(tags[i])

        MODEL = 'bert-base-cased'
        args = TrainingArguments(
            f"{MODEL}-finetuned",
            evaluation_strategy = "epoch",
            save_strategy = "no",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_strategy="no"
        )

        precision = []
        recall = []
        f1 = []
        for train, test in KFold(10, shuffle=True).split(X, y):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            encoded_train, labels_train = tokenize(tokenizer, X_train.tolist(), y_train.tolist())
            encoded_test, labels_test = tokenize(tokenizer, X_test.tolist(), y_test.tolist())
            print(encoded_test[0])
            #tokenizer = RobertaTokenizerFast.from_pretrained(MODEL, add_prefix_sapce=True, truncation=True, is_split_into_words=True)
            #encoded_train, labels_train, unique_tags = tokenize_and_align_labels(tokenizer, X_train.tolist(), y_train.tolist())
            #encoded_test, labels_test, _ = tokenize_and_align_labels(tokenizer, X_test.tolist(), y_test.tolist())

            #data_collator = DataCollatorForTokenClassification(tokenizer)
            model = RobertaCRF.from_pretrained(MODEL)#, num_labels=len(unique_tags))
            trainer = Trainer(
                model,
                args,
                compute_metrics=compute_metrics_token,
                train_dataset=TorchData(encoded_train, labels_train),
                eval_dataset=TorchData(encoded_test, labels_test),
                #data_collator=data_collator,
                tokenizer=tokenizer
            )
            trainer.train()
            eval_metrics = trainer.evaluate()
            precision.append(eval_metrics['eval_precision'])
            recall.append(eval_metrics['eval_recall'])
            f1.append(eval_metrics['eval_f1'])

        print(f'Ensemble Roberta CRF Model {i}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 score: {f1}')

        precision_all.append(np.mean(precision, axis=0))
        recall_all.append(np.mean(recall, axis=0))
        f1_all.append(np.mean(f1, axis=0))

    print(f'Ensemble Roberta CRF Models')
    print(f'Precision scores: {precision_all}')
    print(f'Recall scores: {recall_all}')
    print(f'F1 scores: {f1_all}')


if __name__ == '__main__':
    main()