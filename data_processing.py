from torch.utils.data import Dataset
import numpy as np
from random import shuffle

cat_map = {'Unbalanced_power_relations':0, 'Shallow_solution':1, 
            'Presupposition':2, 'Authority_voice':3, 'Metaphors':4,
            'Compassion':5, 'The_poorer_the_merrier':6}


class TorchData(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


#using the 'span' of PCL to train
def span_data(text):
    data = text.split('\n')[4:-1]
    shuffle(data)
    
    X = []
    y = []
    for line in data:
        columns = line.split('\t')
        X.append(columns[-3])
        y.append(cat_map[columns[-2]])
    
    return np.array(X), np.array(y)


def multilabel_data(text):
    data = text.split('\n')[4:-1]
    
    X = []
    y = []
    paragraphs_seen = []
    for line in data:
        columns = line.split('\t')
        paragraph_id = columns[0]
        if paragraph_id not in paragraphs_seen:
            paragraphs_seen.append(paragraph_id)
            X.append(columns[2])
            temp = [0,0,0,0,0,0,0]
            temp[cat_map[columns[-2]]] = 1
            y.append(temp)
        else:
            temp = y[-1]
            temp[cat_map[columns[-2]]] = 1
            y[-1] = temp
    
    return np.array(X), np.array(y)


def ensemble_data(text):
    data = text.split('\n')[4:-1]
    data = [[int(x.split('\t')[0]),x] for x in data]
    data = sorted(data, key=lambda x: x[0])
    
    Xs = [[],[],[],[],[],[],[]]
    ys = [[],[],[],[],[],[],[]]
    paragraphs_seen = []
    for line in data:
        columns = line[1].split('\t')
        paragraph_id = columns[0]
        label = cat_map[columns[-2]]
        if paragraph_id not in paragraphs_seen:
            paragraphs_seen.append(paragraph_id)
            for X in Xs:
                X.append(columns[2])
            for i,y in enumerate(ys):
                if i == label:
                    y.append(1)
                else:
                    y.append(0)
        else:
            ys[label][-1] = 1
    
    return Xs, ys


def train_and_tags(text):
    data = text.split('\n')[4:-1]
    data = [[int(x.split('\t')[0]),x] for x in data]
    data = sorted(data, key=lambda x: x[0])
    
    Xs = [[],[],[],[],[],[],[]]
    paragraphs_seen = []
    char_tags = [[],[],[],[],[],[],[]]
    tags = [[],[],[],[],[],[],[]]
    for line in data:
        columns = line[1].split('\t')
        span_start = int(columns[5])
        span_end = int(columns[6])
        paragraph = columns[2]
        label = cat_map[columns[-2]]
        paragraph_id = columns[0]
        if paragraph_id not in paragraphs_seen:
            paragraphs_seen.append(paragraph_id)
            if len(char_tags[0]) > 0:
                for i,x_char_tags in enumerate(char_tags):
                    x_char_tags = ''.join(x_char_tags).split(' ')
                    temp_tags = []
                    prev_tag = 'O'
                    for word in x_char_tags:
                        curr_tag = word[0]
                        if curr_tag != 'O':
                            if prev_tag == 'O':
                                temp_tags.append('B-'+curr_tag)
                            else:
                                temp_tags.append('I-'+curr_tag)
                        else:
                            temp_tags.append('O')
                        prev_tag = curr_tag
                    tags[i].append(temp_tags)
                char_tags = [[],[],[],[],[],[],[]]
            for i,character in enumerate(paragraph):
                if character == ' ':
                    [x.append(' ') for x in char_tags]
                else:
                    if i >= span_start and i < span_end:
                        [x.append(str(label)) if label == i else x.append('O') for i,x in enumerate(char_tags)]
                    else:
                        [x.append('O') for x in char_tags]
            for X in Xs:
                X.append(paragraph)
        else:
            for i,character in enumerate(paragraph):
                if i >= span_start and i < span_end and character != ' ':
                    char_tags[label][i] = str(label)
                    
    for i,x_char_tags in enumerate(char_tags):
        x_char_tags = ''.join(x_char_tags).split(' ')
        temp_tags = []
        prev_tag = 'O'
        for word in x_char_tags:
            curr_tag = word[0]
            if curr_tag != 'O':
                if prev_tag == 'O':
                    temp_tags.append('B-'+curr_tag)
                else:
                    temp_tags.append('I-'+curr_tag)
            else:
                temp_tags.append('O')
            prev_tag = curr_tag
        tags[i].append(temp_tags)

    return Xs, tags