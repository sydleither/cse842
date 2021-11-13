import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import KFold
from transformers import BertPreTrainedModel, BertModel

from data_processing import TorchData, ensemble_data
from metrics import compute_metrics_binary

cat_map = {'Unbalanced_power_relations':0, 'Shallow_solution':1, 
               'Presupposition':2, 'Authority_voice':3, 'Metaphors':4,
               'Compassion':5, 'The_poorer_the_merrier':6}


def main():
    pcl = open('data/dontpatronizeme_categories.tsv').read() #TODO some sentences have <h>, not sure if mistake
    Xs, ys = ensemble_data(pcl)

    precision_all = []
    recall_all = []
    f1_all = []
    for i in range(len(ys)):
        X = np.array(Xs[i])
        y = np.array(ys[i])

        MODEL = 'bert-base-uncased'
        args = TrainingArguments(
            f"{MODEL}-finetuned",
            evaluation_strategy = "epoch",
            save_strategy = "no",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
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
            encoded_train = tokenizer(X_train.tolist(), truncation=True)
            encoded_test = tokenizer(X_test.tolist(), truncation=True)

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

        print(f'Ensemble BERT Model {i}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 score: {f1}')

        precision_all.append(np.mean(precision, axis=0))
        recall_all.append(np.mean(recall, axis=0))
        f1_all.append(np.mean(f1, axis=0))

    print(f'Ensemble BERT Models')
    print(f'Precision scores: {precision_all}')
    print(f'Recall scores: {recall_all}')
    print(f'F1 scores: {f1_all}')


if __name__ == '__main__':
    main()