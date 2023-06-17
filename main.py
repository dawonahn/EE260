import datetime
import pdb
import chess
import chess.engine
import chess.pgn
import random
import numpy as np

# fom stockfish import Stockfish
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

PATH=r'/home/dahn017/courses/ee260/stockfish'
DATE=datetime.datetime.now().strftime("%y_%m_%_d")



# Define the Deep Q-Network (DQN) model
class DQN(nn.Module):
    # def __init__(self, input_size, output_size):
    #     super(DQN, self).__init__()
    #     self.fc1 = nn.Linear(input_size, 64)
    #     self.fc2 = nn.Linear(64, 32)
    #     self.fc3 = nn.Linear(32, output_size)

    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     print(x.shape)
    #     return x

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = x.view(-1, 6, 8, 8)  # Reshape the input
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the Deep Q-Learning agent
class DQNAgent:
    def __init__(self, input_size, output_size,
                learning_rate, weight_decay, discount_factor, epsilon, C):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.model = DQN(input_size, output_size)
        self.model_tar = DQN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.action_to_index()
        self.training_step = 0
        self.C = C
        self.train_writer = SummaryWriter(log_dir='tensorboard/chess')


    def action_to_index(self):
        action_to_index = {}
        index_to_action = {}
        index = 0
        for move in chess.SQUARES:
            for target in chess.SQUARES:
                action = chess.Move(move, target)
                uci = action.uci()
                action_to_index[uci] = index
                index_to_action[index] = uci
                index += 1
        self.act_dct = action_to_index
        self.idx_dct = index_to_action
        self.loss_lst = []

    def get_action(self):
        pass
    def get_state(self, board):
        '''
        convert the chess board state
        into a numerical representation (one-hot encoding)
        Pawn  : 1    Rook  : 4
        Night : 2    Queen : 5
        Bishop: 3    King  : 6
        '''

        state = torch.zeros((64, 6))
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_index = chess.PieceType(piece.piece_type)
                color_index = int(piece.color)
                if color_index == 0: color_index = -1
                state[square][piece_index-1] = color_index
                # White is 1 and black is zero 
        return state.flatten()
    
    def check_legal_moves(self, board):
        # Create a mapping from each move to its index
        all_possible_moves = [move.uci() for move in board.legal_moves]
        self.move_mapping = {move: index for index, move in enumerate(all_possible_moves)}
        return all_possible_moves

    def move_to_index(self, move):
        return self.move_mapping[move]

    def select_action(self, state, legal_moves):
        '''
        The select_action method selects an action
        based on the current state and exploration strategy (epsilon-greedy).
        It either chooses a random action with probability epsilon or
        selects the action with the highest Q-value from the model.
        '''
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        # state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_values = self.model(state)
        legal_moves_indices = [self.move_to_index(move) for move in legal_moves]
        # pdb.set_trace()
        valid_q_values = q_values[0][legal_moves_indices]
        max_q_value = torch.max(valid_q_values)
        best_moves = [move for i, move in enumerate(legal_moves) if valid_q_values[i] == max_q_value]
        return random.choice(best_moves)
        # return torch.argmax(valid_q_values)

    def train(self, replay_buffer, batch_size):
        self.training_step +=1
        if len(replay_buffer) < batch_size:
            return
        batch = random.sample(replay_buffer, batch_size)
        # states = torch.tensor([trns[0] for trns in batch], dtype=torch.float)
        states = torch.vstack([trns[0] for trns in batch])
        actions = torch.tensor([self.act_dct[trns[1]] for trns in batch], dtype=torch.long)
        rewards = torch.tensor([trns[2] for trns in batch], dtype=torch.float)
        next_states = torch.vstack([trns[3] for trns in batch])

        # pdb.set_trace()
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_tar_values = self.model_tar(next_states).max(1)[0].detach()
        target_q_values = rewards + self.discount_factor * q_tar_values

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_writer.add_scalar('loss', loss, self.episode)
        # self.loss_lst.append(loss.item())
        print(f"Loss: {loss.item():.4f}")

        if self.training_step % self.C == 0:
            self.model_tar.load_state_dict(self.model.state_dict())
            self.model_tar.eval()


    def play(self, num_episodes, engine_level):
        replay_buffer = []
        episode_moves = []
        for episode in range(num_episodes):
            print("Episode:", episode + 1)
            pgn_moves = []
            total_moves = 0
            self.episode = episode
            board = chess.Board()
            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    legal_moves = self.check_legal_moves(board)
                    state = self.get_state(board)
                    action = self.select_action(state, legal_moves)
                    move = chess.Move.from_uci(action)
                    pgn_moves.append(board.san(move)) # Collect PGN  
                    board.push_uci(action) # Make the move
                    episode_moves.append((state, action))
                    total_moves+=1
                else:
                    engine = chess.engine.SimpleEngine.popen_uci(PATH)
                    result = engine.play(board, chess.engine.Limit(time=engine_level))
                    pgn_moves.append(board.san(result.move))
                    board.push(result.move)
                    engine.quit()
                    total_moves+=1

                next_state = self.get_state(board)
                reward = 1 if board.result() == "1-0" else -1
                replay_buffer.append((state, action, reward, next_state))
                self.train(replay_buffer, batch_size=32)


                if board.is_game_over():
                    with open(f'./results/{DATE}_double_dqn_exp.txt', 'a') as f:
                        f.write(f"Episode:{episode}\t"
                                f"Total moves:{total_moves}\t"
                                f"{' '.join(pgn_moves)}\n")
                    print("Game over:", ' '.join(pgn_moves))
                    break
        torch.save(self.model.state_dict(), "double_dqn_model.pth")
# Main function
if __name__ == "__main__":
    
    input_size = 64 * 6 # chess board (64) and representaion (1x6)
    output_size = 64 * 64 
    learning_rate = 0.001
    weight_decay = 0.001
    discount_factor = 0.99
    epsilon = 0.1
    num_episodes = 10000
    engine_level = 1
    C = 50 # Copying for target Q model.

    agent = DQNAgent(input_size, output_size, learning_rate, weight_decay, discount_factor, epsilon, C)
    agent.play(num_episodes, engine_level)
