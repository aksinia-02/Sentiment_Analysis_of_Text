import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset

class DatasetLoader:
    def __init__(self, vocab_size=10000, batch_size=64):
        """
        Инициализация загрузчика: задаются параметры словаря, батча и токенизатора.
        """
        self.tokenizer = get_tokenizer('basic_english')  # простой токенизатор
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.vocab = None
        self.dataset = None  # датасет будет загружен отдельно

    def load_dataset(self):
        """
        Загружает датасет GoEmotions и преобразует метки в 3 категории: pos, neg, neu.
        """
        raw_dataset = load_dataset("go_emotions")

        # Группы эмоций
        positive = {'admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement',
                    'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'}
        negative = {'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
                    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'}
        neutral = {'neutral'}

        names = raw_dataset['train'].features['labels'].feature.names

        def map_label(example):
            """
            Определяет основную эмоцию для примера и добавляет поле 'label_text'.
            """
            labels = [names[i] for i in example['labels']]
            if any(l in positive for l in labels):
                example['label_text'] = 'pos'
            elif any(l in negative for l in labels):
                example['label_text'] = 'neg'
            elif any(l in neutral for l in labels):
                example['label_text'] = 'neu'
            else:
                example['label_text'] = 'neu'  # fallback
            return example

        dataset = raw_dataset.map(map_label)
        dataset = dataset.filter(lambda x: x['label_text'] in ['pos', 'neg', 'neu'])  # фильтрация
        self.dataset = dataset

    def yield_tokens(self, data_iter):
        """
        Генератор токенов — выдаёт токены из текста для построения словаря.
        """
        for example in data_iter:
            yield self.tokenizer(example['text'])

    def build_vocab(self):
        """
        Строит словарь по токенам из тренировочной части датасета.
        """
        train_iter = self.dataset['train']
        self.vocab = build_vocab_from_iterator(
            self.yield_tokens(train_iter),
            specials=['<unk>', '<pad>'],  # специальные токены
            max_tokens=self.vocab_size
        )
        self.vocab.set_default_index(self.vocab['<unk>'])  # все неизвестные → <unk>

    def text_pipeline(self, text):
        """
        Преобразует текст в список индексов токенов по словарю.
        """
        return [self.vocab[token] for token in self.tokenizer(text)]

    @staticmethod
    def label_pipeline(label):
        """
        Преобразует строковую метку (pos/neg/neu) в число.
        """
        return {'pos': 1, 'neg': 0, 'neu': 2}.get(label, -1)

    def collate_batch(self, batch):
        """
        Собирает один батч: токенизирует, преобразует метки, дополняет паддингом.
        """
        label_list, text_list = [], []
        for example in batch:
            label_list.append(self.label_pipeline(example['label_text']))
            processed_text = torch.tensor(self.text_pipeline(example['text']), dtype=torch.int64)
            text_list.append(processed_text)

        # выравнивание последовательностей по длине
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