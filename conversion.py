import torch
import pandas as pd
import torch.nn as nn

class CSVToTensor:
    def __init__(self, file_path):
        self.data_frame = pd.read_csv(file_path)
        len_dataset = len(self.data_frame)
        self.game_tensor = torch.zeros((len_dataset, 9), dtype=torch.float32)
        self.prediction_tensor = torch.zeros((len_dataset, 9), dtype=torch.float32)
    
    def csv_to_tensor(self, pos):
        if pos >= len(self.data_frame):
            raise ValueError("Position is greater than the number of data in the dataset")
        
        data_pos = self.data_frame.iloc[pos]
        game_stat = data_pos.values[0:9]
        prediction_stat = data_pos.values[9:18]

        self.game_tensor[pos] = torch.tensor(game_stat, dtype=torch.float32)
        self.prediction_tensor[pos] = torch.tensor(prediction_stat, dtype=torch.float32)

    def tensor_to_view(self, pos):
        if pos >= len(self.data_frame):
            raise ValueError("Position is greater than the number of data in the dataset")

        if torch.equal(self.game_tensor[pos], torch.zeros(9)) or torch.equal(self.prediction_tensor[pos], torch.zeros(9)):
            raise ValueError("No tensor data found at this position")
        
        symbols = {0: ' ', 1: 'x', 2: 'o', 3: 'O'}
        board = []
        for i in range(9):
            if self.game_tensor[pos][i] == 1:
                board.append(1)
            elif self.game_tensor[pos][i] == 2:
                board.append(2)
            elif self.prediction_tensor[pos][i] == 2:
                board.append(3)
            else:
                board.append(0)
        
        print("\nCurrent Game State:")
        for i in range(0, 9, 3):
            print(f"{symbols[board[i]]} | {symbols[board[i+1]]} | {symbols[board[i+2]]}")
            if i < 6:
                print("---------")

    def print_data(self):
        print(self.data_frame)

    def create_all_tensor(self):
        for i in range(len(self.data_frame)):
            self.csv_to_tensor(i)
        return self.game_tensor, self.prediction_tensor
    
    def create_a_dataset(self):
        return torch.utils.data.TensorDataset(self.game_tensor, self.prediction_tensor)

# if __name__ == '__main__':
#     position = 1
#     tensor = CSVToTensor('./Datasets/tic_tac_toe_10_games.csv')    
#     tensor.print_data()
#     tensor.csv_to_tensor(position)
#     print(f"Input : {tensor.game_tensor[position]}")
#     print(f"Output : {tensor.prediction_tensor[position]}")
#     tensor.tensor_to_view(position)
