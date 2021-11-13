import numpy as np
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import KFold
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from data_processing import TorchData, multilabel_data
from metrics import compute_metrics_multilabel

cat_map = {'Unbalanced_power_relations':0, 'Shallow_solution':1, 
               'Presupposition':2, 'Authority_voice':3, 'Metaphors':4,
               'Compassion':5, 'The_poorer_the_merrier':6}


#https://github.com/chkla/multilabel-transformer
class MultilabelBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

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

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss() # new loss 📉
                loss = loss_fct(logits.view(-1), labels.view(-1).type_as(logits)) # new loss 📉

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main():
    pcl = open('data/dontpatronizeme_categories.tsv').read() #TODO some sentences have <h>, not sure if mistake
    X, y = multilabel_data(pcl)
    
    MODEL = 'bert-base-uncased'
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

        model = MultilabelBert.from_pretrained(MODEL, num_labels=7)
        trainer = Trainer(
            model,
            args,
            compute_metrics=compute_metrics_multilabel,
            train_dataset=TorchData(encoded_train, y_train),
            eval_dataset=TorchData(encoded_test, y_test),
            tokenizer=tokenizer
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        precision.append(eval_metrics['eval_precision'])
        recall.append(eval_metrics['eval_recall'])
        f1.append(eval_metrics['eval_f1'])

    print('Multilabel BERT Model')
    print(f'Precision: {np.mean(precision, axis=0)}')
    print(f'Recall: {np.mean(recall, axis=0)}')
    print(f'F1 score: {np.mean(f1, axis=0)}')


if __name__ == '__main__':
    main()