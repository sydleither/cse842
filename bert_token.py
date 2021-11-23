import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from sklearn.model_selection import KFold

from data_processing import TorchData, train_and_tags

cat_map = {'Unbalanced_power_relations':0, 'Shallow_solution':1, 
               'Presupposition':2, 'Authority_voice':3, 'Metaphors':4,
               'Compassion':5, 'The_poorer_the_merrier':6}


tag2id = {}

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
def tokenize_and_align_labels(tokenizer, X, tags, keep_word_split=True):
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

    return tokenized_inputs, labels


def main():
    pcl = open('data/dontpatronizeme_categories.tsv').read()
    Xs, tags = train_and_tags(pcl)

    precision_all = []
    recall_all = []
    f1_all = []
    for i in range(len(tags)):
        X = np.array(Xs[i])
        y = np.array(tags[i])

        MODEL = 'distilbert-base-uncased'
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
            encoded_train, labels_train = tokenize_and_align_labels(tokenizer, X_train.tolist(), y_train.tolist())
            encoded_test, labels_test = tokenize_and_align_labels(tokenizer, X_test.tolist(), y_test.tolist())

            data_collator = DataCollatorForTokenClassification(tokenizer)
            model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=3)
            trainer = Trainer(
                model,
                args,
                compute_metrics=compute_metrics_token,
                train_dataset=TorchData(encoded_train, labels_train),
                eval_dataset=TorchData(encoded_test, labels_test),
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            trainer.train()
            eval_metrics = trainer.evaluate()
            precision.append(eval_metrics['eval_precision'])
            recall.append(eval_metrics['eval_recall'])
            f1.append(eval_metrics['eval_f1'])

        print(f'Ensemble BERT Token Model {i}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 score: {f1}')

        precision_all.append(np.mean(precision, axis=0))
        recall_all.append(np.mean(recall, axis=0))
        f1_all.append(np.mean(f1, axis=0))

    print(f'Ensemble BERT TOken Models')
    print(f'Precision scores: {precision_all}')
    print(f'Recall scores: {recall_all}')
    print(f'F1 scores: {f1_all}')


if __name__ == '__main__':
    main()