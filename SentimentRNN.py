import torch
import torch.nn as nn


class SentimentRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        # Вызываем конструктор родительского класса nn.Module, чтобы инициализировать базовую функциональность PyTorch
        super(SentimentRNN, self).__init__()

        # Слой эмбеддингов: преобразует индексы слов в плотные векторы фиксированной размерности
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM-слой: обрабатывает последовательности текста, учитывая их временную зависимость
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)

        # Полносвязный слой: преобразует последнее скрытое состояние RNN в предсказания для трёх классов
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout-слой: случайным образом обнуляет часть элементов для предотвращения переобучения
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Преобразуем входные индексы слов в эмбеддинги и применяем dropout для регуляризации
        embedded = self.dropout(self.embedding(x))

        # Пропускаем эмбеддинги через LSTM, получаем выходные состояния и последние скрытые состояния
        output, (hidden, cell) = self.rnn(embedded)

        # Берём скрытое состояние последнего слоя LSTM и применяем dropout
        hidden = self.dropout(hidden[-1])

        # Пропускаем скрытое состояние через полносвязный слой, чтобы получить предсказания
        return self.fc(hidden)