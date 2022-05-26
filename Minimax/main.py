from games import TicTacToe

if __name__ == "__main__":
    game = TicTacToe((4, 3))
    game.play(use_ab_pruning = False, verbose = True)