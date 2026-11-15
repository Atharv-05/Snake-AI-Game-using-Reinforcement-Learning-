# Snake AI with Deep Q-Learning (PyCharm edition)

This is a PyTorch + Pygame implementation of a Snake-playing agent trained with
Deep Q-Learning (DQN). It is based on the uploaded project and retains its
11-feature state representation, 3 discrete actions, 256-unit hidden layer,
experience replay, epsilon-greedy exploration and Bellman Q-learning update.

## 1. Requirements
- Python 3.10 or 3.11 recommended
- PyCharm
- Packages in `requirements.txt`: PyTorch, Pygame, NumPy and Matplotlib

## 2. Open in PyCharm
1. Extract this ZIP to a normal folder (do not run it from inside the ZIP).
2. In PyCharm choose **Open** and select the `snake-ai-pytorch-main` folder.
3. Open **Settings > Project > Python Interpreter** and create a new virtual
   environment using Python 3.10 or 3.11.
4. Open PyCharm's Terminal. Confirm the terminal is in the folder containing
   `agent.py`, `game.py`, and `model.py`.
5. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   For PyTorch, if pip cannot find a compatible wheel, use the official
   PyTorch installation selector for your OS, Python version and CPU/GPU.

## 3. Train the AI
In the PyCharm terminal:

```bash
python agent.py
```

This trains for 500 complete games by default and opens a live score plot.
To train for longer:

```bash
python agent.py --games 2000
```

To disable the live plot:

```bash
python agent.py --games 500 --no-plot
```

The best-scoring model is saved to `model/model.pth` whenever a new record is
achieved. Training output prints the game number, score and record.

## 4. Play Snake manually
Run `snake_game_human.py` from PyCharm. Use the arrow keys to control the snake.
Close the game window to exit.

## 5. Project files
- `agent.py`: state construction, epsilon-greedy policy, replay memory and loop.
- `game.py`: Pygame environment, reward assignment, collision and movement.
- `model.py`: neural network and Q-learning optimizer.
- `helper.py`: live training graph.
- `snake_game_human.py`: human-playable Snake game.
- `arial.ttf`: font asset loaded by the game.
- `requirements.txt`: Python dependencies.

## 6. Important implementation notes
- State: 11 binary features (danger straight/right/left, current direction,
  and food direction).
- Actions: 3 one-hot actions (straight, turn right, turn left).
- Rewards: +10 for eating food, -10 for collision or timeout, 0 otherwise.
- Network: fully connected 11 -> 256 -> 3, with ReLU on the hidden layer.
- Learning: one-step Q-learning target with discount factor gamma = 0.9.
- Replay memory: maximum 100,000 transitions; batch size 1,000.
- Optimizer: Adam with learning rate 0.001.
- Exploration: epsilon = 80 - number of completed games.

The uploaded source does not include a pre-trained model or saved training
history. A model file is created only after the agent achieves a new record.
Training results depend on random initialization and the runtime environment.

## 7. Troubleshooting
- `ModuleNotFoundError`: activate the PyCharm project interpreter and run
  `python -m pip install -r requirements.txt`.
- `No file 'arial.ttf' found`: set PyCharm's Working directory to the folder
  containing `agent.py` and `arial.ttf`.
- Pygame window does not appear: run on a local desktop session, not a headless
  terminal or remote environment without display support.
- PyTorch installation errors: install the build matching your Python and
  operating system from https://pytorch.org/get-started/locally/.
- If training is slow, start with `python agent.py --games 100`.

## 8. Validation status
The Python files are syntax-checked in the prepared package. Runtime training
requires installing the dependencies and launching Pygame on your machine.
No training score or model performance is claimed until you run the training.
