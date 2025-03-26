import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# ---------------------------
# Sudoku Generation Functions
# ---------------------------
def is_valid(board, row, col, num):
    # Check row
    if any(board[row][x] == num for x in range(9)):
        return False
    # Check column
    if any(board[x][col] == num for x in range(9)):
        return False
    # Check 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    return True

def fill_board(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                numbers = list(range(1, 10))
                random.shuffle(numbers)
                for num in numbers:
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if fill_board(board):
                            return True
                        board[i][j] = 0
                return False
    return True

def generate_complete_sudoku():
    board = [[0] * 9 for _ in range(9)]
    fill_board(board)
    return board

def make_puzzle(board, removals=40):
    # Make a deep copy of board and remove cells
    puzzle = [row[:] for row in board]
    count = removals
    while count > 0:
        i = random.randint(0, 8)
        j = random.randint(0, 8)
        if puzzle[i][j] != 0:
            puzzle[i][j] = 0
            count -= 1
    return puzzle

def flatten_board(board):
    return [cell for row in board for cell in row]

# ---------------------------
# Global Variables & Model Setup
# ---------------------------
reward_history = []  # To store training accuracy over epochs
X_train = []         # List to accumulate training puzzles
y_train = []         # List to accumulate training solutions

# Using a MultiOutputClassifier to treat each of 81 cells as a separate classification task.
# We subtract 1 from the solution digits so that targets are in 0-8.
model = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42))

# ---------------------------
# Tkinter UI Update Functions
# ---------------------------
def update_sudoku_preview(board):
    """Update the 9x9 grid labels with the given board values."""
    for i in range(9):
        for j in range(9):
            value = board[i][j]
            sudoku_labels[i][j].config(text=str(value) if value != 0 else "")

def update_reward_chart():
    """Update the matplotlib chart to plot reward history."""
    ax.clear()
    ax.plot(range(1, len(reward_history) + 1), reward_history, marker='o')
    ax.set_title("Training Accuracy Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    canvas.draw()

# ---------------------------
# Training Process Function
# ---------------------------
def training_process(progress_callback, log_callback):
    total_epochs = 20  # Total training epochs for demonstration
    batch_size = 10    # Number of new puzzles added per epoch
    global X_train, y_train, reward_history, model

    for epoch in range(1, total_epochs + 1):
        # Generate a batch of training samples
        for _ in range(batch_size):
            solution = generate_complete_sudoku()
            puzzle = make_puzzle(solution, removals=40)
            X_train.append(flatten_board(puzzle))
            # Targets: subtract 1 to have classes 0-8
            y_train.append([num - 1 for row in solution for num in row])
        
        # Convert lists to numpy arrays
        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)

        # Inside your training_process():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train_np, y_train_np)
        
        # Train (re-fit) the model on the accumulated dataset
        model.fit(X_train_np, y_train_np)
        
        # Evaluate training accuracy on the whole dataset
        preds = model.predict(X_train_np)
        accuracy = (preds == y_train_np).sum() / y_train_np.size
        
        # Log the training accuracy as a reward
        reward_history.append(accuracy)
        progress_percentage = epoch / total_epochs * 100
        progress_callback(progress_percentage)
        log_callback(f"Epoch {epoch}/{total_epochs}: Training Accuracy = {accuracy:.3f}\n")
        
        # Update the Sudoku preview with the last generated puzzle (reshape flat list into 9x9)
        last_puzzle = X_train[-1]
        last_puzzle_grid = [last_puzzle[i * 9:(i + 1) * 9] for i in range(9)]
        root.after(0, update_sudoku_preview, last_puzzle_grid)
        
        # Update the reward (accuracy) label
        root.after(0, lambda r=accuracy: reward_label.config(text=f"Training Accuracy: {r:.3f}"))
        
        # Update the reward chart
        root.after(0, update_reward_chart)
        
        time.sleep(0.5)  # Simulate delay per epoch
    
    log_callback("Training complete.\n")
    root.after(0, start_button.config, {"state": "normal"})

# ---------------------------
# Tkinter UI Control Functions
# ---------------------------
def start_training():
    start_button.config(state='disabled')
    # Run the training process in a separate thread so the UI remains responsive.
    training_thread = threading.Thread(target=training_process, args=(update_progress, update_log))
    training_thread.start()

def update_progress(percentage):
    progress_bar['value'] = percentage
    root.update_idletasks()

def update_log(message):
    text_area.insert(tk.END, message)
    text_area.see(tk.END)
    root.update_idletasks()

# ---------------------------
# Tkinter UI Setup
# ---------------------------
root = tk.Tk()
root.title("Sudoku ML Training with scikit‑learn")

# Frame for controls
control_frame = tk.Frame(root)
control_frame.pack(pady=5)

start_button = tk.Button(control_frame, text="Start Training", command=start_training)
start_button.pack(side=tk.LEFT, padx=5)

# Progress Bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=5)

# Reward Label to show training accuracy
reward_label = tk.Label(root, text="Training Accuracy: N/A")
reward_label.pack(pady=5)

# Frame for Sudoku Preview (9x9 grid)
preview_frame = tk.Frame(root)
preview_frame.pack(pady=5)

sudoku_labels = []
for i in range(9):
    row_labels = []
    for j in range(9):
        label = tk.Label(preview_frame, text="", width=3, height=1, borderwidth=1,
                         relief="solid", font=("Arial", 12))
        label.grid(row=i, column=j, padx=1, pady=1)
        row_labels.append(label)
    sudoku_labels.append(row_labels)

# Matplotlib chart for Reward Progress
fig, ax = plt.subplots(figsize=(4, 3))
ax.set_title("Training Accuracy Over Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=5)
canvas.draw()

# Log Text Area
text_area = tk.Text(root, height=10, width=50)
text_area.pack(pady=5)

root.mainloop()
