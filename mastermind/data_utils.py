import numpy as np
from .board import Board
import random
from . import mastermind_utils


def create_data(filename, num_samples):
    # Shape of data created:
    # Shape of x:  [num_samples, 234]
    # Shape of y_: [num_samples, 4]
    x, y_ = [], []

    for i in range(num_samples):
        if i % 1000 == 0:
            print(str(i // 1000) + 'K datapoints created.')

        # To create a random training datum, start by choosing the color code:
        code = [random.randrange(0, 6) for _ in range(4)]

        # Start with an empty board
        board = Board.create_empty_board()

        # Fill the board with pins, as if num_moves moves were already done.
        # Don't train for num_moves = 0; that move shouldn't be done by the neural net but be random.
        num_moves = random.randrange(1, 9)
        fill_board_randomly(board, code, num_moves)

        x.append(board.to_input_for_network())
        y_.append(code)

    print('Saving created data in numpy file...')
    np.save(filename + '_inputs', np.array(x))
    np.save(filename + '_outputs', np.array(y_))
    print('create_data is done.')


def fill_board_randomly(board, code, num_moves):
    for row in range(num_moves):  # row is a row on the board (in each move, one row is filled with pins)
        # Try a random code:
        code_guess = [random.randrange(0, 6) for _ in range(4)]
        board.big_pins[row] = code_guess
        # Find the amount of black and whites for this code and code_guess
        blacks, whites = mastermind_utils.count_blacks_and_whites(code, code_guess)
        board.small_pins[row] = [blacks, whites]


create_data('training_data_1M', 1000000)
