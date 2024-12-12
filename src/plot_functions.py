import matplotlib.pyplot as plt
import pandas as pd

def plot_perplexity(results, ratios):
    perplexities = [result["perplexity"]["test_ppl"] for result in results]

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

PROMPT = """
    A young girl named Lila discovers an ancient book in the attic of her family home. 
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