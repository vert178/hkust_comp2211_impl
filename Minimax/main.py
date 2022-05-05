from minimax import Game
import numpy as np

# A basic tic tac toe implementation
class TicTacToe(Game):
    def __init__(self, size = (3,3), winning_line_len = 3):
        # Assume for simplicity, you are always x and the AI is always o regardless of who goes first
        super().__init__(list(range(1, size[0] * size[1] + 1)), size, print_lookup = {0: " ", 1: "x", -1: "o"})
        self.wll = winning_line_len
    
    def check_score(self):
        # Check horizontal lines
        for i in range(self.size[0]):
            for j in range(self.size[1] - self.wll + 1):
                isLine = True
                if self.state[i, j] == 0: 
                    continue
                for k in range(1, self.wll):
                    if self.state[i, j] != self.state[i, j + k]:
                        isLine = False
                        break
                if isLine:
                    return -self.state[i, j]
        # Check vertial lines
        for i in range(self.size[0] - self.wll + 1):
            for j in range(self.size[1]):
                isLine = True
                if self.state[i, j] == 0: 
                    continue
                for k in range(1, self.wll):
                    if self.state[i, j] != self.state[i + k, j]:
                        isLine = False
                        break
                if isLine:
                    return -self.state[i, j]
        # Check \ diagnoal lines
        for i in range(self.size[0] - self.wll + 1):
            for j in range(self.size[1] - self.wll + 1):
                isLine = True
                if self.state[i, j] == 0: 
                    continue
                for k in range(1, self.wll):
                    if self.state[i, j] != self.state[i + k, j + k]:
                        isLine = False
                        break
                if isLine:
                    return -self.state[i, j]
        # Check / diagonal lines
        for i in range(self.wll - 1, self.size[0]):
            for j in range(self.size[1] - self.wll + 1):
                isLine = True
                if self.state[i, j] == 0: 
                    continue
                for k in range(1, self.wll):
                    if self.state[i, j] != self.state[i - k, j + k]:
                        isLine = False
                        break
                if isLine:
                    return -self.state[i, j]        
        # No lines have been found. If the board is full, then its draw nd return 0, otherwise return 2
        empty_spaces = np.count_nonzero(self.state == 0)
        if empty_spaces == 0: return 0
        return 2
    
    def make_move(self, move: int, is_AIs_turn: bool):
        if type(move) != type(0): return False
        if move not in self.valid_moves: return False
        row = (move - 1) // self.size[1]
        col = (move - 1) % self.size[1]
        if self.state[row, col] != 0:
            return False
        self.state[row, col] = -1 if is_AIs_turn else 1
        return True
    
    def undo_move(self, move: int, is_AIs_turn: bool):
        if type(move) != type(0): return False
        if move not in self.valid_moves: return False
        row = (move - 1) // self.size[1]
        col = (move - 1) % self.size[1]
        expected_piece = -1 if is_AIs_turn else 1
        if self.state[row, col] != expected_piece:
            return False
        self.state[row, col] = 0
        return True


if __name__ == "__main__":
    game = TicTacToe((3,3))
    game.play()