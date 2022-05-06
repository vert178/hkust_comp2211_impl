import numpy as np
import time

# This provides a template for a game that we can do minimax algorithm
class Game:
    def __init__(self, valid_moves: list, size: tuple, print_sep_len: int = 2, print_lookup: dict = {}):
        # self.size is a tuple storing (rows, columns)
        # self.valid_moves is a list of ints of all possible moves
        # self.print_sep_length determines the print board behaviour
        self.size = (int(size[0]), int(size[1]))
        assert len(self.size) == 2 and self.size[0] > 0 and self.size[1] > 0 
        self.state = np.zeros(self.size, dtype = int)
        self.print_lookup = print_lookup
        self.valid_moves = valid_moves
        self.print_sep_len = print_sep_len

    def print_board(self):
        # Print separators on odd rows else print game stuff according to the print lookup table
        for i in range(2 * self.size[0] - 1):
            for j in range(2 * self.size[1] - 1):
                if i % 2 == 0:
                    state = self.print_lookup[self.state[i // 2, j // 2]] if j % 2 == 0 else "|"
                    print(state.ljust(self.print_sep_len), end = "")
                else:
                    print("-" * self.print_sep_len, end = "")
            print()
        print()
    
    def check_score(self):
        # Returns 0 if it is a draw, 1 if the AI wins, -1 if the player wins, and 2 if the game is not finished
        # This to make sure the check win function is implemented before we start
        raise NotImplementedError
    
    def make_move(self, move: int, is_AI_turn: bool):
        # Make sure the move is an integer and is not obviously invalid
        # Return True if the move is valid, and make the move, otherwise return False and nothing happens
        raise NotImplementedError
    
    def undo_move(self, move: int, is_AI_turn: bool):
        # Return True if the move is valid, and undos the move, otherwise return False and nothing happens
        raise NotImplementedError
    
    # Returns the best move using minimax algorithm
    def solve(self, ab_pruning = True, max_depth = 1e99):
        t1 = time.time()
        sc, move, num_moves = 0, 0, 0
        if ab_pruning:
            sc, move, num_moves = self.minimax_ab_pruning(True, -1e99, 1e99, 0, max_depth)
        else:
            sc, move, num_moves = self.minimax_recursive(True, 0, max_depth)
        t2 = time.time()
        t = f"{round(t2-t1, 5)} seconds"
        return sc, move, num_moves, t

    def minimax_recursive(self, is_AIs_turn, depth, max_depth = 15):
    # returns the score of the move and the best move and the number of moves considered
        # Initialize the score to +inf or -inf, and best move
        # Check the current state. If it is not finished:
        #   For each possibly valid move: try move
        #   If it is valid:
        #       Make said move
        #       Do minimax on the moved game board with incremented depth and reversed state
        #       Calculate the score of the move. If it improves the score, then update best_move; and then update score
        #       Undo the move
        #   return score of the board
        # Else:
        #   return the win state of the board
        current_state, score, best_move, moves_considered = self.check_score(), -1e99 if is_AIs_turn else 1e99, -1, 1
        if current_state == 2 and depth <= max_depth:
            for move in self.valid_moves:
                if self.make_move(move, is_AIs_turn):
                    child_score, bm, num_moves = self.minimax_recursive(not is_AIs_turn, depth + 1, max_depth)
                    moves_considered += num_moves
                    ns = max(score, child_score) if is_AIs_turn else min(score, child_score)
                    if ns != score: best_move = move
                    score = ns
                    assert self.undo_move(move, is_AIs_turn)
            return score, best_move, moves_considered
        else: return current_state, best_move, 1

    def minimax_ab_pruning(self, is_AI_turn, alpha, beta, depth, max_depth = 15):
        # Initialize score, moves, and check the current state of the game
        # If the game have not finished then
        # For each possible move:
        #   Try to make the move and if it is valid:
        #       Pass alpha and beta to the child node and perform recursion magic
        #       Undo the move
        #       update score and alpha/beta
        #       if alpha >= beta, then prune all the remaining trees and exit early
        #   return score and best move
        # Otherwise return score and stuff
        current_state, score, best_move, num_moves_considered = self.check_score(), -1e99 if is_AI_turn else 1e99, -1, 1
        if current_state == 2 and depth <= max_depth:
            for move in self.valid_moves:
                if self.make_move(move, is_AI_turn):
                    child_score, child_best_move, num_moves = self.minimax_ab_pruning(not is_AI_turn, alpha, beta, depth + 1, max_depth)
                    assert self.undo_move(move, is_AI_turn)
                    num_moves_considered += num_moves
                    # If is ai's turn then we want to do maximization
                    if is_AI_turn:
                        new_score = alpha = max(score, child_score)
                        if new_score > score:
                            best_move = move
                        score = new_score
                    else:
                        new_score = beta = min(score, child_score)
                        if new_score < score:
                            best_move = move
                        score = new_score
                    if alpha >= beta:
                        break
            return score, best_move, num_moves_considered
        else:
            return current_state, best_move, 1

        
    # Maybe try hashing a few of the starting positions to decrease search time in the future


    def play(self, human_starts_first = True, verbose = False, use_ab_pruning = True):
        sep_line = "\n" + "=" * 40 + "\n"  
        is_AI_turn = human_starts_first

        # Main loop
        score = self.check_score()
        while score == 2:
            # Change the player and print the board
            is_AI_turn = not is_AI_turn
            print(sep_line)
            self.print_board()
            print()

            # Prompt to make move
            if is_AI_turn:
                print("It is the AI's turn. Please enter an integer to make a move: ", end = "")
                sc, move, num_moves, t = self.solve(use_ab_pruning)
                print(move)
                self.make_move(move, True)
                if verbose: print("Current state = ", score, " Best possible outcome = ", sc, " Number of moves considered = ", num_moves, " Time taken = ", t)
            else:
                move = input("It is your turn. Please enter an integer to make a move: ")
                while not self.make_move(int(move), False):
                    move = input("This is not a valid move. Please enter an integer to make a move: ")
            # Update score
            score = self.check_score()
            
        
        # The game has finished
        print(sep_line)
        self.print_board()
        if score == 0:
            print("The game has ended. It is a draw.")
        if score == 1:
            print("The game has ended. The AI won. Don't worry that is normal")
        if score == -1:
            print("The game has ended. Wow you won! That means the code has a bug!")