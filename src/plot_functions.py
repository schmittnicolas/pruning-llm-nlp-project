import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metric_data, ratios, metric_name):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        ratios, metric_data, marker="o", linestyle="-", color="b", label=metric_name
    )

    # Adding labels and title
    plt.xlabel("Pruning Ratio (%)", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f"{metric_name} vs. Pruning Ratio", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Display the plot
    plt.show()


PROMPT = """A young girl named Lila discovers an ancient book in the attic of her family home. 
    The book is said to contain powerful secrets, but it is written in a language no one can understand…
    """


def compare_prompt(results, columns, prompt=PROMPT):
    print(prompt.replace("\n", "").strip())
    columns = [str(c) for c in columns]

    text_generation = [
        result["text_generation"]["generated_text"]
        .replace(prompt, "")
        .replace("\n", "")
        for result in results
    ]

    data = pd.DataFrame([text_generation])
    data.columns = columns

    pd.set_option("display.max_colwidth", None)

    styled_data = data.style.set_properties(
        **{"text-align": "center", "vertical-align": "middle"}
    ).set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])

    return styled_data


def compare_ecological_impact(results, columns):
    energy_consumption_kwh = [
        result["ecological_impact"]["energy_consumption_joules"] for result in results
    ]
    co2_emissions_kg = [
        result["ecological_impact"]["co2_emissions_grams"] for result in results
    ]

    data = pd.DataFrame(
        {
            "Energy Consumption (J)": energy_consumption_kwh,
            "CO2 Emissions (g)": co2_emissions_kg,
        }
    )

    data.index = columns
    return data


def plot_metrics_horizontal(ratios, perplexity, model_size):
    # Create subplots for a horizontal layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)  # Horizontal layout
    plt.subplots_adjust(wspace=0.3)  # Adjust space between subplots

    # Perplexity plot
    axes[0].plot(
        ratios, perplexity, marker="o", linestyle="-", color="blue", label="Perplexity"
    )
    axes[0].set_title("Perplexity vs. Pruning Ratio", fontsize=14)
    axes[0].set_xlabel("Pruning Ratio (%)", fontsize=12)
    axes[0].set_ylabel("Perplexity", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Model size plot
    axes[1].plot(
        ratios, model_size, marker="^", linestyle="-.", color="red", label="Model Size"
    )
    axes[1].set_title("Memory Usage vs. Pruning Ratio", fontsize=14)
    axes[1].set_xlabel("Pruning Ratio (%)", fontsize=12)
    axes[1].set_ylabel("Memory Usage (MB)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    # Show the plots
    plt.tight_layout()
    plt.show()


###############################################################################################
# STRUCURED PRUNING PLOT                                                                      #
###############################################################################################

def plot_structured_evaluation_metrics(ratios, metrics):
    """
    Create a multi-plot visualization of key metrics.
    
    Args:
        ratios (list): Pruning ratios
        metrics (dict): Dictionary of metrics to plot
    """
    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Metrics Across Pruning Ratios', fontsize=16)
    
    # Metrics to plot
    plot_configs = [
        ('Model Size (MB)', 'blue', 'o'),
        ('Perplexity', 'green', '^'),
        ('Inference Time (s)', 'red', 's'),
        ('CO2 Emissions (g)', 'purple', 'D')
    ]
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Plot each metric
    for (metric, color, marker), ax in zip(plot_configs, axes_flat):
        ax.plot([r*100 for r in ratios], metrics[metric], 
                color=color, marker=marker, linestyle='-')
        ax.set_title(metric)
        ax.set_xlabel('Pruning Ratio (%)')
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# display_structured_results(results, ratios)