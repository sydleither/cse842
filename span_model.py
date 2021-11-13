import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import KFold

from data_processing import TorchData, span_data
from metrics import compute_metrics

cat_map = {'Unbalanced_power_relations':0, 'Shallow_solution':1, 
               'Presupposition':2, 'Authority_voice':3, 'Metaphors':4,
               'Compassion':5, 'The_poorer_the_merrier':6}


def main():
    pcl = open('data/dontpatronizeme_categories.tsv').read() #TODO some sentences have <h>, not sure if mistake
    X, y = span_data(pcl)

    MODEL = 'bert-base-uncased'
    args = TrainingArguments(
        f"{MODEL}-finetuned",
        evaluation_strategy = "epoch",
        save_strategy = "no",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_strategy="no"
    )

    precision = []
    recall = []
    f1 = []
    for train, test in KFold(10).split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        encoded_train = tokenizer(X_train.tolist())
        encoded_test = tokenizer(X_test.tolist())

        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=7)
        trainer = Trainer(
            model,
            args,
            compute_metrics=compute_metrics,
            train_dataset=TorchData(encoded_train, y_train),
            eval_dataset=TorchData(encoded_test, y_test),
            tokenizer=tokenizer
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        precision.append(eval_metrics['eval_precision'])
        recall.append(eval_metrics['eval_recall'])
        f1.append(eval_metrics['eval_f1'])

    print('Span BERT Model')
    print(f'Precision: {np.mean(precision, axis=0)}')
    print(f'Recall: {np.mean(recall, axis=0)}')
    print(f'F1 score: {np.mean(f1, axis=0)}')


if __name__ == '__main__':
    main()