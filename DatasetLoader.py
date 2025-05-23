import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

class DatasetLoader:
    def __init__(self, dataset_name='IMDB', vocab_size=10000, batch_size=64):
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.vocab = None
        self.dataset_name = dataset_name


    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)


    def build_vocab(self):
        train_iter = IMDB(split='train')
        self.vocab = build_vocab_from_iterator(self.yield_tokens(train_iter), specials=['<unk>', '<pad>'],
                                               max_tokens=self.vocab_size)
        self.vocab.set_default_index(self.vocab['<unk>'])


    def text_pipeline(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]


    def label_pipeline(self, label):
        return {'pos': 2, 'neg': 0, 'neu': 1}[label] if label in ['pos', 'neg', 'neu'] else 1


    def collate_batch(self, batch):
        label_list, text_list = [], []
        for label, text in batch:
            label_list.append(self.label_pipeline(label))
            processed_text = torch.tensor(self.text_pipeline(text), dtype=torch.int64)
            text_list.append(processed_text)
        text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=self.vocab['<pad>'])
        return torch.tensor(label_list, dtype=torch.long), text_list


    def get_data_loader(self, split='train'):
        data_iter = IMDB(split=split)
        return DataLoader(data_iter, batch_size=self.batch_size, collate_fn=self.collate_batch)


