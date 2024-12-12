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
    energy_consumption_kwh = [result["ecological_impact"]['energy_consumption_kwh'] for result in results]
    co2_emissions_kg = [result["ecological_impact"]['inference_energy_kwh'] for result in results]


    data = pd.DataFrame({
    'Energy Consumption (kWh)': energy_consumption_kwh,
    'CO2 Emissions (kg)': co2_emissions_kg
    })

    # Set the index names as 'Energy Consumption' and 'CO2 Emissions'
    data.index = ['Energy Consumption', 'CO2 Emissions']


    data.columns = columns



    return data