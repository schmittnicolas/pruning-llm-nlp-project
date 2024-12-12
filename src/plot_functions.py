import matplotlib.pyplot as plt

def plot_perplexity(perplexities, ratios):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, perplexities, marker='o', linestyle='-', color='b', label='Perplexity')

    # Adding labels and title
    plt.xlabel("Pruning Ratio (%)", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    plt.title("Perplexity vs. Pruning Ratio", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Display the plot
    plt.show()