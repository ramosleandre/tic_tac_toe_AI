import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, random_split
from conversion import CSVToTensor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_size = 9
        self.output_size = 9
        self.hidden_size = 2*27

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.crossloss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data = None
        self.val_data = None

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    def load_data(self, file_path):
        dataloader = CSVToTensor(file_path)
        dataloader.create_all_tensor()
        dataset = dataloader.create_a_dataset()
    
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_data, self.val_data = random_split(dataset, [train_size, val_size])
        self.train_data = DataLoader(self.train_data, batch_size=16, shuffle=True)
        self.val_data = DataLoader(self.val_data, batch_size=16, shuffle=True)

    def train_model(self, epochs):
        self.to(self.device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for src, trg in self.train_data:
                src = src.to(self.device)
                trg = trg.to(self.device)
                self.optimizer.zero_grad()
                output = self.forward(src)
                loss = self.crossloss(output, trg.argmax(dim=1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.train_data)
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    model = Model()
    # choose the dataset file path
    model.load_data('./Datasets/tic_tac_toe_500_games.csv')
    # choose the number of epochs
    model.train_model(100)
