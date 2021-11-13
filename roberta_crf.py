import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import KFold
from transformers import BertPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torchcrf import CRF

from data_processing import TorchData, ensemble_data
from metrics import compute_metrics_binary

cat_map = {'Unbalanced_power_relations':0, 'Shallow_solution':1, 
               'Presupposition':2, 'Authority_voice':3, 'Metaphors':4,
               'Compassion':5, 'The_poorer_the_merrier':6}


#https://github.com/shushanxingzhe/transformers_ner
class RobertaCRF(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = RobertaModel(config, add_pooling_layer=False)
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


def main():
    pcl = open('data/dontpatronizeme_categories.tsv').read() #TODO some sentences have <h>, not sure if mistake
    Xs, ys = ensemble_data(pcl)

    precision_all = []
    recall_all = []
    f1_all = []
    for i in range(len(ys)):
        X = np.array(Xs[i])
        y = np.array(ys[i])

        MODEL = 'roberta-base'
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
            encoded_train = tokenizer(X_train.tolist(), truncation=True)
            encoded_test = tokenizer(X_test.tolist(), truncation=True)

            model = RobertaCRF.from_pretrained(MODEL, num_labels=2)
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

        print(f'Ensemble Roberta Model {i}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 score: {f1}')

        precision_all.append(np.mean(precision, axis=0))
        recall_all.append(np.mean(recall, axis=0))
        f1_all.append(np.mean(f1, axis=0))

    print(f'Ensemble Roberta Models')
    print(f'Precision scores: {precision_all}')
    print(f'Recall scores: {recall_all}')
    print(f'F1 scores: {f1_all}')


if __name__ == '__main__':
    main()