import torch

from game import Connect2Game
from model import Connect2Model
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 64,
    'numIters': 500,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 100,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 2,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
}

game = Connect2Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect2Model(board_size, action_size, device)

trainer = Trainer(game, model, args)
trainer.learn()
