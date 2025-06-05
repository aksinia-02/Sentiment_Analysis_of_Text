import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
from tqdm import tqdm

class DatasetLoader:
    def __init__(self, vocab_size=10000, batch_size=64):
        """
        Инициализация загрузчика: задаются параметры словаря, батча и токенизатора.
        """
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.vocab = None
        self.dataset = None
        self.load_dataset()

    def load_dataset(self):
        """
        Загружает датасет DynaSent из локальных файлов и фильтрует метки positive, negative, neutral.
        """
        data_files = {
            "train": "dynasent-v1.1/dynasent-v1.1-round01-yelp-train.jsonl",
            "validation": "dynasent-v1.1/dynasent-v1.1-round01-yelp-dev.jsonl",
            "test": "dynasent-v1.1/dynasent-v1.1-round01-yelp-test.jsonl"
        }
        self.dataset = load_dataset("json", data_files=data_files)
        self.dataset = self.dataset.filter(lambda x: x['gold_label'] in ['positive', 'negative', 'neutral'], desc="Filtering dataset")
        print(f"Train size: {len(self.dataset['train'])}, Validation size: {len(self.dataset['validation'])}, Test size: {len(self.dataset['test'])}")

    def yield_tokens(self, data_iter):
        """
        Генератор токенов — выдаёт токены из текста для построения словаря.
        """
        for example in data_iter:
            yield self.tokenizer(example['sentence'])

    def build_vocab(self):
        """
        Строит словарь по токенам из тренировочной части датасета.
        """
        train_iter = self.dataset['train']
        self.vocab = build_vocab_from_iterator(
            self.yield_tokens(train_iter),
            specials=['<unk>', '<pad>'],
            max_tokens=self.vocab_size
        )
        self.vocab.set_default_index(self.vocab['<unk>'])

    def text_pipeline(self, text):
        """
        Преобразует текст в список индексов токенов по словарю.
        """
        return [self.vocab[token] for token in self.tokenizer(text)]

    @staticmethod
    def label_pipeline(label):
        """
        Преобразует строковую метку (positive/negative/neutral) в число.
        """
        return {'positive': 1, 'negative': 0, 'neutral': 2}[label]

    def collate_batch(self, batch):
        """
        Собирает один батч: токенизирует, преобразует метки, дополняет паддингом.
        """
        label_list, text_list = [], []
        for example in batch:
            label_list.append(self.label_pipeline(example['gold_label']))
            processed_text = torch.tensor(self.text_pipeline(example['sentence']), dtype=torch.int64)
            text_list.append(processed_text)

        text_list = nn.utils.rnn.pad_sequence(
            text_list, batch_first=True, padding_value=self.vocab['<pad>']
        )

        return torch.tensor(label_list, dtype=torch.long), text_list

    def get_data_loader(self, split='train'):
        """
        Возвращает DataLoader, который загружает данные батчами с подготовкой.
        """
        return DataLoader(
            self.dataset[split],
            batch_size=self.batch_size,
            collate_fn=self.collate_batch
        )