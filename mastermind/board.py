import numpy as np


class Board:
    def __init__(self, big_pins, small_pins):
        # big_pins: shape = [9, 4] (9 = #rows, 4 = #holes), each element = color in [0,6], WHERE 0 MEANS NO COLOR)
        self.big_pins = big_pins
        # small_pins: shape = [9, 2] (9 = #rows, 2 = black & white), each element = #pins of that color in [0,4])
        self.small_pins = small_pins

    def to_input_for_network(self):
        """ Converts the data in this Board object (big_pins, small_pins) to an array of shape [234],
        suitable as one element of the input tensor for the network. """
        result = []
        for row in range(9):
            for hole in range(4):
                # Add one-hot vector for this hole (- 1 because each element is in [0,6] and not [0,6))
                result.extend([int(i == self.big_pins[row][hole] - 1) for i in range(6)])
            result.append(self.small_pins[row][0])  # No. of black pins
            result.append(self.small_pins[row][1])  # No. of white pins
        return result

    @staticmethod
    def create_empty_board():
        return Board(np.zeros([9, 4]), np.zeros([9, 2]))
