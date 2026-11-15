import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    """Update the training chart in a normal desktop Matplotlib window."""
    if not plt.get_fignums():
        plt.figure("Snake AI training", figsize=(10, 5))

    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_title("Deep Q-Learning - Snake training")
    ax.set_xlabel("Completed games")
    ax.set_ylabel("Score")
    ax.plot(scores, label="Score per game", alpha=0.55)
    ax.plot(mean_scores, label="Running mean score", linewidth=2)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.pause(0.001)
