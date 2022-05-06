from games import TicTacToe

if __name__ == "__main__":
    game = TicTacToe((4,4))
    game.play(use_ab_pruning = True, verbose = True)