import matplotlib.pyplot as plt
import pandas as pd



def plot_metrics(metric_data, ratios, metric_name):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, metric_data, marker='o', linestyle='-', color='b', label=metric_name)

    # Adding labels and title
    plt.xlabel("Pruning Ratio (%)", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f"{metric_name} vs. Pruning Ratio", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Display the plot
    plt.show()




PROMPT = """A young girl named Lila discovers an ancient book in the attic of her family home. 
    The book is said to contain powerful secrets, but it is written in a language no one can understand…
    """

def compare_prompt(results, columns, prompt=PROMPT):
    print(prompt)
    text_generation = [result["text_generation"]['generated_text'].replace(prompt, '').replace('\n', '') for result in results]

    data = pd.DataFrame([text_generation], index=[0])
    data.columns = columns

    pd.set_option('display.max_colwidth', None)

    styled_data = data.style.set_properties(**{
        'text-align': 'center',
        'vertical-align': 'middle'
    }).set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]}
    ])

    return styled_data


def compare_ecological_impact(results, columns):
    energy_consumption_kwh = [result["ecological_impact"]['energy_consumption_joules'] for result in results]
    co2_emissions_kg = [result["ecological_impact"]['co2_emissions_grams'] for result in results]


    data = pd.DataFrame({
    'Energy Consumption (kWh)': energy_consumption_kwh,
    'CO2 Emissions (kg)': co2_emissions_kg
    })

    # Set the index names as 'Energy Consumption' and 'CO2 Emissions'
    data.index = ['Energy Consumption', 'CO2 Emissions']


    data.columns = columns



    return data



def plot_metrics_vertical(ratios, perplexity, flops, model_size):
    # Create subplots with a shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    plt.subplots_adjust(hspace=0.3)  # Adjust space between subplots

    # Perplexity plot
    axes[0].plot(ratios, perplexity, marker='o', linestyle='-', color='blue', label='Perplexity')
    axes[0].set_title("Perplexity vs. Pruning Ratio", fontsize=14)
    axes[0].set_ylabel("Perplexity", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)

    # FLOPs plot
    axes[1].plot(ratios, flops, marker='s', linestyle='--', color='green', label='FLOPs')
    axes[1].set_title("FLOPs vs. Pruning Ratio", fontsize=14)
    axes[1].set_ylabel("FLOPs", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)

    # Model size plot
    axes[2].plot(ratios, model_size, marker='^', linestyle='-.', color='red', label='Model Size')
    axes[2].set_title("Memory Usage vs. Pruning Ratio", fontsize=14)
    axes[2].set_xlabel("Pruning Ratio (%)", fontsize=12)
    axes[2].set_ylabel("Memory Usage (MB)", fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(fontsize=10)

    # Show the plots
    plt.tight_layout()
    plt.show()