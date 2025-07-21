import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_success_rate(file_path):
    """
    Calculates the success rate from an aggregated attack results file.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at '{file_path}'")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. The file might be empty or malformed.")
        return None
    if not isinstance(data, list):
        print(f"Error: Expected a list of attacks in '{file_path}', but found {type(data)}.")
        return None
    total_attacks = len(data)
    if total_attacks == 0:
        return 0.0, 0, 0
    successful_attacks = sum(1 for attack in data if attack.get("is_changed") is True)
    success_rate = (successful_attacks / total_attacks) * 100
    return success_rate, successful_attacks, total_attacks


def create_small_multiples_plot(all_results, k_values, attack_methods_order):
    """
    Generates a Small Multiples plot (faceted bar charts) for attack success rates.
    This function has been updated to display bar charts instead of line plots.
    """
    sns.set_style("whitegrid")

    # Custom color mapping
    color_map = {
        "Heuristic-Remove": "#4A4D7D",  # Dark blue/purple
        "RL-Remove": "#5E8D88",      # Teal/blue-green
        "RL-Add(CAIRAGE)": "#7DB66E", # Medium green
        "RL-Add(Semantic)": "#B0D87E" # Lighter green/lime
    }

    # Determine layout for subplots
    n_methods = len(attack_methods_order)
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols # Calculate rows needed

    # Create the figure and a grid of subplots, sharing X and Y axes for consistency
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    # Set the overall title for the plot
    fig.suptitle('Attack Success Rate vs. Budget (k)',
                 fontsize=18, fontweight='bold', color='black', y=0.96)

    # Define bar width relative to the spacing between k values
    bar_width = 0.8

    for i, method_name in enumerate(attack_methods_order):
        ax = axes[i]
        k_data = sorted(all_results[method_name].keys())
        rates = [all_results[method_name][k] for k in k_data]

        # Plot the bars for the current method
        ax.bar(k_data, rates,
               width=bar_width,
               color=color_map.get(method_name, 'gray'), # Use custom color
               alpha=0.8) # Slightly transparent bars

        # Add data labels for each bar
        for k, rate in zip(k_data, rates):
            ax.annotate(f'{rate:.1f}%',
                        xy=(k, rate),
                        xytext=(0, 5), # Offset text slightly above the bar
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=8,
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.7))

        # Set the title for each subplot (attack method name)
        ax.set_title(method_name, fontsize=12, fontweight='bold', color='darkslategray')
        ax.set_ylim(0, 100) # Ensure consistent Y-axis limits
        ax.set_xticks(k_values) # Set X-ticks
        ax.set_xticklabels([f"k={k}" for k in k_values], fontsize=10) # Label X-ticks

        ax.grid(True, linestyle='--', alpha=0.7) # Add grid

    # Set common Y-axis label for the entire figure
    fig.text(0.02, 0.5, 'Success Rate (%)', va='center', rotation='vertical',
             fontsize=14, fontweight='bold', color='dimgray')

    # Set common X-axis label for the entire figure
    fig.text(0.5, 0.02, 'Attack Budget (k)', ha='center',
             fontsize=14, fontweight='bold', color='dimgray')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0.03, 0.05, 1, 0.96])
    plot_filename = "attack_success_rates_small_multiples_bar.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\n📈 Analysis complete. Small Multiples bar plot saved to '{plot_filename}'")
    plt.show()


if __name__ == '__main__':
    ATTACK_FILES = {
        "Heuristic-Remove": "/home/neha2022/thesis/outputs_thesis_main/baseline_attack_outputs/",
        "RL-Remove": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_remove/",
        "RL-Add(CAIRAGE)": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_add_cairage/",
        "RL-Add(Semantic)": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_add_semantic/"
    }

    k_values_to_process = [3, 5, 7, 9, 11, 13, 15]

    all_results = {}
    print("Success Rate Analysis")
    for method_name, base_path in ATTACK_FILES.items():
        print(f"\nAnalyzing Method: {method_name}")
        method_results = {}
        for k in k_values_to_process:
            file_path = os.path.join(base_path, f'aggregated_attacks_k{k}.json')
            analysis_result = calculate_success_rate(file_path)
            if analysis_result:
                rate, _, _ = analysis_result
                method_results[k] = rate
        all_results[method_name] = method_results

    if not any(all_results.values()):
        print("\nNo results to plot.")
    else:
        ordered_methods = ["Heuristic-Remove", "RL-Remove", "RL-Add(CAIRAGE)", "RL-Add(Semantic)"]
        create_small_multiples_plot(all_results, k_values_to_process, ordered_methods)
