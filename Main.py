import torch
import torch.nn as nn
import torch.optim as optim
from DatasetLoader import DatasetLoader
from SentimentRNN import SentimentRNN
from tqdm import tqdm

class Main:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = "sentiment_RNNmodel.pth"
        self.vocab_size = 10000
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.output_dim = 3  # pos, neg, neu
        self.n_layers = 2
        self.dropout = 0.5
        self.epochs = 5
        self.loader = DatasetLoader(vocab_size=self.vocab_size)
        self.loader.build_vocab()

    def train(self):
        train_loader = self.loader.get_data_loader(split='train')
        model = SentimentRNN(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim,
            self.n_layers,
            self.dropout
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        try:
            model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                # Добавляем tqdm для прогресс-бара
                for labels, texts in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch"):
                    labels, texts = labels.to(self.device), texts.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(texts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.4f}")
        except KeyboardInterrupt:
            print("\n⛔ Training interrupted by user.")
        finally:
            torch.save(model.state_dict(), self.model_path)
            print(f"\n💾 Model saved to {self.model_path}")

    def load_model(self):
        model = SentimentRNN(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim,
            self.n_layers,
            self.dropout
        ).to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def predict(self, text):
        model = self.load_model()
        indices = torch.tensor([self.loader.text_pipeline(text)], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = model(indices)
            pred = output.argmax(1).item()
            label_map = {1: "positive", 0: "negative", 2: "neutral"}
            return label_map.get(pred, "unknown")

if __name__ == "__main__":
    main = Main()
    main.train()
    print("Predictions:")
    print(f"I am so happy today! -> {main.predict('I am so happy today!')}")
    print(f"This is the worst. -> {main.predict('This is the worst.')}")
    print(f"Okay. -> {main.predict('Okay.')}")