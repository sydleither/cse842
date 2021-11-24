from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
import torch
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score

from data_processing import multilabel_data, add_backtranslation_multilabel


#https://colab.research.google.com/drive/1M5Qx-FVJYNqFdvpJgggIZaWk5SpGS1Nu?usp=sharing#scrollTo=zyBcJoHtJHE2
def main():
    train = open('data/train.tsv').read()
    test = open('data/test.tsv').read()

    X_train, y_train = multilabel_data(train)
    X_test, y_test = multilabel_data(test)

    X_train, y_train = add_backtranslation_multilabel(X_train, y_train)

    train_df = pd.DataFrame(list(zip(X_train, y_train)), columns=['text', 'labels'])

    task2_model_args = MultiLabelClassificationArgs(num_train_epochs=10,
                                                no_save=True, 
                                                no_cache=True, 
                                                overwrite_output_dir=True
                                                )
    task2_model = MultiLabelClassificationModel("roberta", 
                                                'roberta-base', 
                                                num_labels=7,
                                                args = task2_model_args, 
                                                use_cuda=True)

    task2_model.train_model(train_df)
    preds, _ = task2_model.predict(X_test)

    precision = precision_score(np.array(y_test), preds, average=None)
    recall = recall_score(np.array(y_test), preds, average=None)
    f1 = f1_score(np.array(y_test), preds, average=None)

    print(f'RoBERTa Baseline')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')

    ran = [[random.choice([0,1]) for k in range(7)] for k in range(0,len(y_test))]
    precision = precision_score(np.array(y_test), np.array(ran), average=None)
    recall = recall_score(np.array(y_test), np.array(ran), average=None)
    f1 = f1_score(np.array(y_test), np.array(ran), average=None)

    print(f'Random Baseline')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')


if __name__ == '__main__':
    main()