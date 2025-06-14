import torch
import torch.nn as nn
import torch.optim as optim
from DatasetLoader import DatasetLoader
from SentimentRNN import SentimentRNN
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


class Main:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "sentiment_RNNmodel.pth"
        self.vocab_size = 10000
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.output_dim = 3  # pos, neg, neu
        self.n_layers = 2
        self.dropout = 0.5
        self.epochs = 6
        self.loader = DatasetLoader(vocab_size=self.vocab_size)
        self.loader.build_vocab()

    def plot_metrics(self, train_losses, val_losses, val_accuracies, val_f1s):
        """
        Plots and saves the training/validation loss, accuracy, and F1 score over epochs.
        """
        epochs = range(1, self.epochs + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, label="Validation Accuracy")
        plt.plot(epochs, val_f1s, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Accuracy and F1 over Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig("performance_graph.png")
        plt.show()

    def train(self):
        # Fine-tuning: load model weights if they exist, otherwise start from scratch
        train_loader = self.loader.get_data_loader(split="train")
        model = SentimentRNN(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim,
            self.n_layers,
            self.dropout,
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        # Lists to store metrics for plotting
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_f1s = []
        try:
            model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for labels, texts in tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch"
                ):
                    labels, texts = labels.to(self.device), texts.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(texts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)
                train_losses.append(avg_loss)
                print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
                # Валидация после каждой эпохи
                val_loss, val_acc, val_f1 = self.validate(model, criterion)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_f1s.append(val_f1)
                print(
                    f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}"
                )
        except KeyboardInterrupt:
            print("\n⛔ Training interrupted by user.")
        finally:
            torch.save(model.state_dict(), self.model_path)
            print(f"\n💾 Model saved to {self.model_path}")
            self.plot_metrics(train_losses, val_losses, val_accuracies, val_f1s)
        return model

    def validate(self, model, criterion):
        """
        Оценивает модель на валидационной выборке, возвращает среднюю потерю, точность и F1.
        """
        val_loader = self.loader.get_data_loader(split="validation")
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for labels, texts in tqdm(val_loader, desc="Validating", unit="batch"):
                labels, texts = labels.to(self.device), texts.to(self.device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, accuracy, f1

    def test(self, model):
        """
        Оценивает модель на тестовой выборке, возвращает среднюю потерю, точность и F1.
        """
        test_loader = self.loader.get_data_loader(split="test")
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for labels, texts in tqdm(test_loader, desc="Testing", unit="batch"):
                labels, texts = labels.to(self.device), texts.to(self.device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, accuracy, f1

    def load_model(self):
        model = SentimentRNN(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.output_dim,
            self.n_layers,
            self.dropout,
        ).to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def predict(self, text):
        model = self.load_model()
        indices = torch.tensor([self.loader.text_pipeline(text)], dtype=torch.long).to(
            self.device
        )
        with torch.no_grad():
            output = model(indices)
            pred = output.argmax(1).item()
            label_map = {1: "positive", 0: "negative", 2: "neutral"}
            return label_map.get(pred, "unknown")


if __name__ == "__main__":
    main = Main()
    print("Training model...")
    trained_model = main.train()
    print("Evaluating on test set...")
    test_loss, test_acc, test_f1 = main.test(trained_model)
    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}"
    )
    print("Predictions:")
    print(f"I am so happy today! -> {main.predict('I am so happy today!')}")
    print(f"This is the worst. -> {main.predict('This is the worst.')}")
    print(f"Okay. -> {main.predict('Okay.')}")