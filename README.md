# Tic-Tac-Toe AI

Welcome to the Tic-Tac-Toe AI project! This repository aims to develop an intelligent agent capable of playing Tic-Tac-Toe and predicting the next move to play. The project involves data collection, model training, and evaluation to create a robust AI that can compete against human players or other AI opponents.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Simulation](#data-simulation)


## Introduction

Tic-Tac-Toe is a simple yet classic game played on a 3x3 grid. Players take turns marking a cell in the grid with their symbol (x or o), aiming to place three of their marks in a horizontal, vertical, or diagonal row. Despite its simplicity, Tic-Tac-Toe serves as an excellent starting point for developing and testing AI algorithms.

This project is designed to create an AI that can (for now):
- Simulate games and collect data.
- Train on game data to predict the best next move.
- Train with the hand to understand what really do a training model

## Features

- **Game Simulation**: Automatically simulate and record multiple games of Tic-Tac-Toe.
- **Data Processing**: Convert game records into tensors for machine learning models.
- **Debugging function for each step**: Understand each layer of the model with vizualisation of tensor and his values 

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- pandas
- csv
- random

### Clone the Repository

```sh
git clone https://github.com/ramosleandre/Tic_Tac_Toe_AI.git
cd Tic_Tac_Toe_AI
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

## Usage

### Data Simulation

Simulate games and generate a CSV file with game states and moves.

```python
import csv
import random

# Add the simulate_games function here
# Your simulation code from above

# Run the simulation for N games
simulate_games(1000)
```

### Convert CSV to Tensors

Convert the simulated game data into tensors for model training.
choose your dataset and for debug, the line at debug.

```python

tensor_converter = CSVToTensor('./Datasets/tic_tac_toe_10_games.csv')
tensor_converter.csv_to_tensor(position)
print(f"Input : {tensor_converter.game_tensor[position]}")
print(f"Output : {tensor_converter.prediction_tensor[position]}")
```
Output :
```python
Input : tensor([1., 0., 1., 2., 0., 0., 0., 0., 0.])
Output : tensor([0., 2., 0., 0., 0., 0., 0., 0., 0.])
```
Convert you tensor (position in your CSV) in a vizualizer
```python

tensor_converter.tensor_to_view(position)
```
Output :
```bash
Current Game State:
x | O | x
---------
o |   |  
---------
  |   |
```
