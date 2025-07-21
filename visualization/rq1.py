import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_rates_and_efficiency(file_path, k):
    """
    Calculates the success rate and attack efficiency from an aggregated attack results file.
    Efficiency is defined as (Success Rate / k).
    """
    if not os.path.exists(file_path):
        return 0.0, 0, 0, 0.0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. The file might be empty or malformed.")
        return 0.0, 0, 0, 0.0
    if not isinstance(data, list):
        print(f"Error: Expected a list of attacks in '{file_path}', but found {type(data)}.")
        return 0.0, 0, 0, 0.0

    total_attacks = len(data)
    if total_attacks == 0:
        return 0.0, 0, 0, 0.0

    successful_attacks = sum(1 for attack in data if attack.get("is_changed") is True)
    success_rate = (successful_attacks / total_attacks) * 100

    # Calculate efficiency
    efficiency = success_rate / k if k > 0 else 0.0

    return success_rate, successful_attacks, total_attacks, efficiency


def plot_absolute_successful_attacks(all_results, k_values):
    """
    Plots the absolute number of successful attacks as a line plot.
    (This function remains unchanged as the focus is on efficiency plot)
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("viridis", len(all_results.keys()))

    for i, (method_name, results) in enumerate(all_results.items()):
        successful_counts = [results.get(k, {}).get('successful', 0) for k in k_values]
        ax.plot(k_values, successful_counts, marker='o', linestyle='-', label=method_name, color=colors[i])

    ax.set_ylabel('Number of Successfully Attacked QA-Pairs', fontsize=12)
    ax.set_xlabel('Attack Budget (k)', fontsize=12)
    ax.set_title('Absolute Number of Successful Attacks vs. Budget (k)', fontsize=16, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(title="Attack Method", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig("absolute_successful_attacks.png")
    print("Plot of absolute successful attacks saved.")



def create_efficiency_small_multiples_plot(all_efficiency_results, k_values, attack_methods_order):
    """
    Generates a Small Multiples plot (faceted bar charts) for attack efficiency,
    matching the design and color scheme of the provided reference image.
    """
    sns.set_style("whitegrid")

    # Define plot_titles
    plot_titles = {
        "Heuristic-Remove": "Heuristic-Remove",
        "RL-Remove": "RL-Remove",
        "RL-Add(CAIRAGE)": "RL-Add(CAIRAGE)",
        "RL-Add(Semantic)": "RL-Add(Semantic)"
    }

    color_map = {
            "Heuristic-Remove": "#4A4D7D",  # Dark blue/purple
            "RL-Remove": "#5E8D88",      # Teal/blue-green
            "RL-Add(CAIRAGE)": "#7DB66E", # Medium green
            "RL-Add(Semantic)": "#B0D87E" # Lighter green/lime
        }

    n_methods = len(attack_methods_order)
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle('Attack Efficiency vs. Budget (k)',
                 fontsize=18, fontweight='bold', color='black', y=0.96)

    bar_width = 0.5

    # Determine global Y-axis maximum from all efficiency results to set shared ylim
    max_efficiency = 0
    for method_name in attack_methods_order:
        if method_name in all_efficiency_results:
            method_efficiencies = list(all_efficiency_results[method_name].values())
            if method_efficiencies:
                max_efficiency = max(max_efficiency, max(method_efficiencies))

    y_lim_max = max_efficiency * 1.25
    if y_lim_max < 10:
        y_lim_max = 10

    for i, method_name in enumerate(attack_methods_order):
        ax = axes[i]

        k_data_str = []
        efficiencies = []
        if method_name in all_efficiency_results:
            for k in sorted(all_efficiency_results[method_name].keys()):
                k_data_str.append(f"k={k}")
                efficiencies.append(all_efficiency_results[method_name][k])

        ax.bar(k_data_str, efficiencies,
               width=bar_width,
               color=color_map.get(method_name, 'gray'),
               alpha=0.8)

        for k_label, efficiency in zip(k_data_str, efficiencies):
            ax.annotate(f'{efficiency:.1f}%',
                        xy=(k_label, efficiency),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=8,
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.7))

        ax.set_title(plot_titles[method_name], fontsize=12, fontweight='bold',  color='darkslategray')

#         ax.set_xlabel('Attack Budget (k)', fontsize=10)
        ax.tick_params(axis='x', labelsize=11)

#         if i % n_cols == 0:
#             ax.set_ylabel('Attack Efficiency (Success Rate % / k)', fontsize=12)
#         else:
#             ax.set_ylabel('')

        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(0, y_lim_max)
        ax.set_axisbelow(True)

        ax.grid(axis='y', linestyle='--', alpha=0.7, color='lightgray')
        # Corrected: Replaced 'b=False' with 'visible=False'
        ax.grid(axis='x', linestyle='--', alpha=0.7, color='lightgray')

     # Set common Y-axis label for the entire figure
        fig.text(0.02, 0.5, 'Attack Efficiency (Success Rate % / k)', va='center', rotation='vertical',
                 fontsize=14, fontweight='bold', color='dimgray')

        # Set common X-axis label for the entire figure
        fig.text(0.5, 0.02, 'Attack Budget (k)', ha='center',
                 fontsize=14, fontweight='bold', color='dimgray')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0.03, 0.05, 1, 0.96])
    plot_filename = "attack_efficiency_small_multiples_bar_matched.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Analysis complete. Small Multiples bar plot for efficiency saved to '{plot_filename}'")
    plt.show()


if __name__ == '__main__':
    ATTACK_FILES = {
           "Heuristic-Remove": "/home/neha2022/thesis/outputs_thesis_main/baseline_attack_outputs",
           "RL-Remove": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_remove/",
           "RL-Add(CAIRAGE)": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_add_cairage/",
           "RL-Add(Semantic)": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_add_semantic/"
    }
    k_values_to_process = [3, 5, 7, 9, 11, 13, 15]

    all_efficiency_results = {}
    print("Attack Efficiency Analysis")
    for method_name, base_path in ATTACK_FILES.items():
        print(f"Analyzing Method: {method_name}")
        method_efficiencies = {}
        for k in k_values_to_process:
            file_path = os.path.join(base_path, f'aggregated_attacks_k{k}.json')

            success_rate, successful, total, efficiency = calculate_rates_and_efficiency(file_path, k)

            print(f"k={k}: {efficiency:.2f} efficiency (Success Rate: {success_rate:.2f}% | Successful: {successful}/{total})")

            method_efficiencies[k] = efficiency
        all_efficiency_results[method_name] = method_efficiencies

    ordered_methods = ["Heuristic-Remove", "RL-Remove", "RL-Add(CAIRAGE)", "RL-Add(Semantic)"]

    has_any_data = any(
        any(val > 0 for val in method_data.values())
        for method_data in all_efficiency_results.values() if method_data
    )

    if not has_any_data:
        print("No valid efficiency results to plot (all efficiencies were 0 or files missing). Please check your data files.")
    else:
        create_efficiency_small_multiples_plot(all_efficiency_results, k_values_to_process, ordered_methods)