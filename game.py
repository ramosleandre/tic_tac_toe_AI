import csv
import random

def initialize_board():
    return [0] * 9

def print_board(board):
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    print("\nCurrent Game State:")
    for i in range(0, 9, 3):
        print(f"{symbols[board[i]]} | {symbols[board[i+1]]} | {symbols[board[i+2]]}")
        if i < 6:
            print("---------")

def check_win(board):
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)              # Diagonals
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != 0:
            return True
    return False

def check_tie(board):
    return all(x != 0 for x in board)

def simulate_game(writer):
    board = initialize_board()
    available_moves = list(range(9))
    current_player = 1
    board_state1 = [0] * 9

    while True:
        if not available_moves:
            break
        move = random.choice(available_moves)
        board[move] = current_player

        if current_player == 2:
            board_state2 = [0] * 9
            board_state2[move] = 2
            writer.writerow(board_state1 + board_state2)
            board_state1[move] = 2

        else:
            board_state1[move] = 1

        if check_win(board) or check_tie(board):
            break

        available_moves.remove(move)
        current_player = 2 if current_player == 1 else 1

def simulate_games(num_games):
    with open('tic_tac_toe_10_games.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ['P1_Cell1', 'P1_Cell2', 'P1_Cell3', 'P1_Cell4', 'P1_Cell5', 'P1_Cell6',
                   'P1_Cell7', 'P1_Cell8', 'P1_Cell9', 'P2_Cell1', 'P2_Cell2', 'P2_Cell3',
                   'P2_Cell4', 'P2_Cell5', 'P2_Cell6', 'P2_Cell7', 'P2_Cell8', 'P2_Cell9']
        writer.writerow(headers)
        for _ in range(num_games):
            simulate_game(writer)

# Run the simulation for N games
simulate_games(10)
