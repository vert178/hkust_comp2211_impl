# This provides a template for a game that we can do minimax algorithm
import numpy as np
minimax_iter_count = 0
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
        self.help = "Not Implemented!"

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
    
    # Undos a given move
    def undo_move(self, move: int, is_AI_turn: bool):
        # Return True if the move is valid, and undos the move, otherwise return False and nothing happens
        raise NotImplementedError
    
    # Returns the best move using minimax algorithm
    def solve(self):
        global minimax_iter_count
        score, best_move = self.minimax_recursive(True, 0)
        m = minimax_iter_count
        minimax_iter_count = 0
        return best_move, m, score

    def minimax_recursive(self, is_AIs_turn, depth, max_depth = 15):
        global minimax_iter_count
        minimax_iter_count += 1
    # depth keeps track of the current depth of the recursion, max_depth denotes the max search length that you want to do
    # returns the score of the move and the best move
        # Check the current state. If it is not finished:
        #   Initialize the score to +inf or -inf, and best move
        #   For each possibly valid move: try move
        #   If it is valid:
        #       Make said move
        #       Do minimax on the moved game board with incremented depth and reversed state
        #       Calculate the score of the move. If it improves the score, then update best_move; and then update score
        #       Undo the move
        #   return score of the board
        # Else:
        #   return the win state of the board
        current_state, score, best_move = self.check_score(), -1e99 if is_AIs_turn else 1e99, -1
        if current_state == 2:
            for move in self.valid_moves:
                if self.make_move(move, is_AIs_turn):
                    child_score, bm = self.minimax_recursive(not is_AIs_turn, depth + 1, max_depth)
                    ns = max(score, child_score) if is_AIs_turn else min(score, child_score)
                    if ns != score:
                        best_move = move
                    score = ns
                    self.undo_move(move, is_AIs_turn)
            return score, best_move
        else:
            return current_state, best_move
     
    def play(self, human_starts_first = True):
        sep_line = "\n" + "=" * 40 + "\n"  
        is_AI_turn = human_starts_first
        # Main loop
        score = self.check_score()
        while score == 2:
            # Change the player
            is_AI_turn = not is_AI_turn
            print(sep_line)
            self.print_board()
            print()

            # Prompt to make move
            if is_AI_turn:
                print("It is the AI's turn. Please enter an integer to make a move: ", end = "")
                try:
                    move, m, sc = self.solve()
                except KeyboardInterrupt:
                    global minimax_iter_count
                    print("Moves considered: ", minimax_iter_count)
                print(move)
                print("Number of considered moves = ", m)
                self.make_move(move, True)
            else:
                move = input("It is your turn. Please enter an integer to make a move: ")
                while move == "help" or not self.make_move(int(move), False):
                    if move == "help":
                        print(self.help)
                        move = input("It is your turn. Please enter an integer to make a move: ")
                    else:
                        move = input("This is not a valid move. Please enter an integer to make a move: ")
            # Update score
            score = self.check_score()
            print("Score = ", score)
        
        # The game has finished
        print(sep_line)
        if score == 0:
            print("The game has ended. It is a draw.")
        if score == 1:
            print("The game has ended. The AI won. Don't worry that is normal")
        if score == -1:
            print("The game has ended. Wow you won! That means the code has a bug!")