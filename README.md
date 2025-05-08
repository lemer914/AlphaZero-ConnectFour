# AlphaZero-ConnectFour

This project implements two reinforcement learning approaches—**AlphaZero-inspired (AlphaFour)** and **Deep Q-Network (DQN)**—to train agents that play the game **Connect Four**. The code includes training, evaluation, hyperparameter tuning, and saved models ready for reuse or deployment.

---

## Notebooks

### `AlphaFour.ipynb`
Implements **AlphaFour**, a reinforcement learning agent inspired by the AlphaZero algorithm.

- Defines the Connect Four environment and gameplay mechanics.
- Implements Monte Carlo Tree Search (MCTS) for move selection.
- Has functionality to tune and train a neural network to evaluate game states and suggest optimal moves.
- Uses self-play to iteratively improve the agent's performance.

### `DQN_model_train_and_eval.ipynb`
Trains and evaluates a **Deep Q-Network (DQN)** agent for Connect Four.

- Defines the DQN model architecture and training process.
- Uses experience replay and target networks for stability.
- Evaluates the agent against random or rule-based opponents after training.

### `DQN_model_tuning.ipynb`
Tunes hyperparameters for the DQN agent to improve performance.

- Explores different network architectures.
- Helps identify the best hyperparameter settings for training.

---

## Saved Models

### `alphazero_model.pth`
- Trained model weights for the AlphaFour agent.
- Used with Monte Carlo Tree Search to guide decision-making in Connect Four.
- Can be loaded to evaluate performance, continue training, or play against.

### `connect_four_dqn.pth`
- Trained Deep Q-Network (DQN) model for Connect Four.
- Encodes Q-values for various actions across game states.
- Loadable for evaluation or further training.

---

## Dependencies

Dependencies can be found in `requirements.txt`.
