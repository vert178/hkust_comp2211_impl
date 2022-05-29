import numpy as np
import time

class Connect4:
    # Board size: rows x columns
    def __init__(self, AI_go_first = False, board_size = (6, 7), winning_length = 4, solve_techniques = ["ab-pruning"], verbose = 0):
        # Initialize board with required size
        self.board = np.zeros(board_size, dtype = object)
        self.is_AI_turn = AI_go_first
        self.AI_piece = "x"
        self.human_piece = "o"
        self.winning_positions = self.generate_winning_positions(winning_length, board_size)
        self.available_moves = list(range(board_size[1]))
        self.solve_techniques = solve_techniques
        self.verbose = verbose
        self.moves_considered = 0

    def print_board(self):
        print_sep_len = 3
        for j in range(2 * self.board.shape[1] - 1):
            if j % 2 == 0:
                print(str(j//2).ljust(print_sep_len), end = "")
            else:
                print(" " * print_sep_len, end = "")
        print()
        for j in range(2 * self.board.shape[1] - 1):
            if j % 2 == 0:
                print("↓".ljust(print_sep_len), end = "")
            else:
                print(" " * print_sep_len, end = "")
        print("\n")
        
        for i in range(2 * self.board.shape[0] - 1):
            for j in range(2 * self.board.shape[1] - 1):
                if i % 2 == 0:
                    s = " " if self.board[i//2, j//2] == 0 else self.board[i//2, j//2]
                    state = str(s) if j % 2 == 0 else "|"
                    print(state.ljust(print_sep_len), end = "")
                else:
                    print("-" * print_sep_len, end = "")
            print()
        print()

    # Generate a list of all winning positions for us to check against
    def generate_winning_positions(self, winning_length, board_size):
        rows, cols = board_size
        winning_pos = []

        # Horizontal lines. Loop through all columns and the first (row - winning length) rows
        # Then put in all the horizontal lines of the form (i,j), (i, j+1), etc
        for i in range(rows):
            for j in range(cols - winning_length + 1):
                line = [(i, j+k) for k in range(winning_length)]
                winning_pos.append(line)

        # Vertical lines
        for i in range(rows - winning_length + 1):
            for j in range(cols):
                line = [(i+k, j) for k in range(winning_length)]
                winning_pos.append(line)

        # / diagonal line
        for i in range(winning_length - 1, rows):
            for j in range(cols - winning_length + 1):
                line = [(i-k, j+k) for k in range(winning_length)]
                winning_pos.append(line)
        
        # \ diagonal line
        for i in range(rows - winning_length + 1):
            for j in range(cols - winning_length + 1):
                line = [(i+k, j+k) for k in range(winning_length)]
                winning_pos.append(line)

        return winning_pos

    def is_valid_move(self, move):
        # Sanity checks - make sure move is int between 0 and 7 (number of columns)
        if type(move) != type(0): return False
        if move < 0: return False
        if move >= self.board.shape[1]: return False
        return True


    # Makes a move, flips the state of the turn variable, and returns True if the move is valid and successfully made
    def make_move(self, move, is_AI_turn):
        if not self.is_valid_move(move): return False

        # AI plays 1 and human plays -1
        game_piece = self.AI_piece if is_AI_turn else self.human_piece
        
        # Implements the drop down mechanism in connect 4
        # for(int i = number of columns, i >= 0, i--):
        #   if cell is empty:
        #       put game piece in cell
        #       flips game state
        #       break

        i = self.board.shape[0] - 1
        while i > -1:
            if self.board[i, move] == 0:
                self.board[i, move] = game_piece
                return True
            i -= 1
        return False

    def undo_move(self, move, is_AI_turn):
        if not self.is_valid_move(move): return False

        game_piece = self.AI_piece if is_AI_turn else self.human_piece

        for i in range(self.board.shape[0]):
            if self.board[i, move] == game_piece:
                self.board[i, move] = 0
                return True
            elif self.board[i, move] != 0:
                print("cannot find the correct game piece")
                return False
            i -= 1
        return False


    # Returns 1 if AI wins, 0 if draw, -1 if human wins, and 2 if the game is not finished
    def check_game(self):
        # For positions in winning positions:
        #   remember the state of the first cell
        #   If first cell is empty then stop searching this position
        #   For other positions:
        #       Check state against first cell
        #       if it doesnt match: stop searching this position
        #       If all matches then return according to the state of the first cell (consequently, every cell)
        # If the code arrives here that means there are no lines. Then, return 0 if every cell is filled, else 2

        for line in self.winning_positions:            
            first_cell_state = self.board[line[0]]
            if first_cell_state == 0: continue

            is_match = True
            for cell in line:
                if self.board[cell] != first_cell_state:
                    is_match = False
                    break
            
            if is_match: return 1 if first_cell_state == self.AI_piece else -1
        
        if np.count_nonzero(self.board == 0) > 0: return 2
        return 0
    

    def solve(self, is_AI_turn: bool):
        self.moves_considered = 0
        if "depth" in self.solve_techniques:
            score, move = self.minimax_abd(is_AI_turn, -1e99, 1e99, 0)
        elif "ab-pruning" in self.solve_techniques:
            score, move = self.minimax_ab_pruning(is_AI_turn, -1e99, 1e99)
        else:
            score, move = self.minimax(is_AI_turn)
        return score, move
    
    def minimax(self, is_AI_turn: bool):
        # Initiate score variable, and check the state of the game
        # If it is done:
        #   return state (as an integer -1, 0, 1)
        # For each available move:
        #   Try move
        #   Apply minimax on this moved state (the state is already flipped in the make_move and undo_move methods)
        #   update the score variable according to whoever's turn it is supposed to be
        #   undo the move
        # return the score
        self.moves_considered += 1
        state = self.check_game()
        best_move = -1
        if state != 2:
            return state, 0

        score = -1e99 if is_AI_turn else 1e99
        for move in self.available_moves:
            if not self.make_move(move, is_AI_turn): continue
            if best_move == -1:
                best_move = move
            child_score, child_move = self.minimax(not is_AI_turn)
            if is_AI_turn:
                if child_score > score:
                    best_move = move
                score = max(child_score, score)
            else:
                if child_score < score:
                    best_move = move
                score = min(child_score, score)
            assert self.undo_move(move, is_AI_turn)
        return score, best_move

    def minimax_ab_pruning(self, is_AI_turn, alpha, beta):
        # Initialize score and best move variable
        # if state is 2 then return state and best move
        # for each valid moves:
        #   Try move
        #   pass alpha beta to child and do minimax-ab there
        #   Undo move
        #   if it is AI turn then update score, move and alpha
        #   if it is human turn then update score, move and beta
        #   if alpha >= beta: exit early
        # return score and move
        self.moves_considered += 1
        state, best_move = self.check_game(), -1
        score = -1e99 if is_AI_turn else 1e99
        if state != 2:
            return state, 0
        for move in self.available_moves:
            if not self.make_move(move, is_AI_turn): continue
            if best_move == -1:
                best_move = move
            child_score, child_move = self.minimax_ab_pruning(not is_AI_turn, alpha, beta)
            assert self.undo_move(move, is_AI_turn)
            if is_AI_turn:
                if child_score > score:
                    score = child_score
                    best_move = move
                    alpha = score
            else:
                if child_score < score:
                    score = child_score
                    move = move
                    beta = score
            if alpha >= beta:
                break
        return score, best_move


    def minimax_abd(self, is_AI_turn: bool, alpha: float, beta: float, depth: int):
        self.moves_considered += 1
        epsilon = 1e-3
        # Initialize score and best move variable
        # Since shorter path moves are more favourable we add a depth * epsilon factor to the game trees
        # if the player wins then return -1 + depth * epsilon
        # if the AI wins then return 1 - depth * epsilon
        # Otherwise the state = 2, then for each valid moves:
        #   try move
        #   perform minimax abd on child with +1 depth
        #   undo move
        #   If it is AI's turn then maximize score
        #   If it is human's turn then minimize score
        #   If alpha >= beta: exit early
        # return score and best move
        score, best_move = -1e99 if is_AI_turn else 1e99, -1
        state = self.check_game()
        if state == 0: return 0, 0
        if state == 1: return 1 - depth * epsilon, 0
        if state == -1: return -1 + depth * epsilon, 0
        for move in self.available_moves:
            if not self.make_move(move, is_AI_turn): continue
            if best_move == -1: best_move = move
            child_score, child_move = self.minimax_abd(not is_AI_turn, alpha, beta, depth + 1)
            assert self.undo_move(move, is_AI_turn)
            if is_AI_turn and child_score > score:
                score = child_score
                best_move = move
                alpha = child_score
            elif not is_AI_turn and child_score < score:
                score = child_score
                best_move = move
                beta = child_score
            if alpha >= beta:
                break
        return score, best_move


    def play(self, verbose = 0):
        # While the game is not finished:
        # If it is AI's turn:
        #   solve the board using minimax
        #   let AI make its turn
        #   otherwise

        state = self.check_game()
        while True:
            t = 0
            print("=" * 40 + "\n")
            self.print_board()
            if state != 2:
                break
            
            if self.is_AI_turn:
                t1 = time.time()
                score, move = self.solve(True)
                t2 = time.time()
                t = round(t2 - t1, 5)
                if self.verbose > 1.5: 
                    print("AI makes the move: ", move, " This move has score: ", score)
                if self.verbose > 0.5:
                    print("Moves considered: ", self.moves_considered)
                    print( "Time taken to solve: ", t, " seconds")
                assert self.make_move(move, True)
                self.is_AI_turn = False
            else:
                while True:
                    move = input("Please input a move: ")
                    try:
                        move = int(move)
                        assert self.make_move(move, False)
                        break
                    except:
                        move = print("That was not a valid move.", end = "")
                self.is_AI_turn = True

            # Print debug information
            if self.verbose > 1.5:
                state = self.check_game()
                print("Current state of the game: ", state)
            state = self.check_game()
        
        if state == 1:
            print("The AI won. No suprises right?")
        if state == -1:
            print("Wow you won. Congrats on beating the theoretically impossible AI")
        if state == 0:
            print("Draw. That's cool.")

if __name__ == "__main__":
    c4 = Connect4(AI_go_first=True, board_size=(3, 3), winning_length=3, solve_techniques = ["ab-pruning", "depth"], verbose = 2)
    c4.play()