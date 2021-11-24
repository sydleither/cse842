import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import KFold

from data_processing import TorchData, span_data
from metrics import compute_metrics


def main():
    train = open('data/train.tsv').read()
    test = open('data/test.tsv').read()

    X_train, y_train = span_data(train, True)
    X_test, y_test = span_data(test, False)

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encoded_train = tokenizer(X_train)
    encoded_test = tokenizer(X_test)

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
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')


if __name__ == '__main__':
    main()