import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.utils import shuffle

from data_processing import TorchData, binary_data, add_backtranslation
from metrics import compute_metrics_binary


def main(MODEL, backtranslation=False):
    train = open('data/train.tsv').read()
    test = open('data/test.tsv').read()

    Xs_train, ys_train = binary_data(train)
    Xs_test, ys_test = binary_data(test)

    precision = []
    recall = []
    f1 = []
    for i in range(len(ys_train)):
        X_train = Xs_train[i]
        X_test = Xs_test[i]
        y_train = ys_train[i]
        y_test = ys_test[i]

        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        if backtranslation:
            X_train, y_train = add_backtranslation(X_train, y_train)

        args = TrainingArguments(
            f"{MODEL}-finetuned",
            evaluation_strategy = "epoch",
            save_strategy = "no",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_strategy="no"
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        encoded_train = tokenizer(X_train, truncation=True)
        encoded_test = tokenizer(X_test, truncation=True)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        trainer = Trainer(
            model,
            args,
            compute_metrics=compute_metrics_binary,
            train_dataset=TorchData(encoded_train, y_train),
            eval_dataset=TorchData(encoded_test, y_test),
            tokenizer=tokenizer
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        precision.append(eval_metrics['eval_precision'])
        recall.append(eval_metrics['eval_recall'])
        f1.append(eval_metrics['eval_f1'])

    print(f'Stacked {MODEL} Models')
    print(f'Precision scores: {precision}')
    print(f'Recall scores: {recall}')
    print(f'F1 scores: {f1}')


if __name__ == '__main__':
    main('bert-base-uncased', True)