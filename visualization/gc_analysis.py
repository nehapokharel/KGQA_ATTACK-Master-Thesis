import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

ATTACK_FILES = {
    "Heuristic-Remove": "/home/neha2022/thesis/outputs_thesis_main/baseline_attack_outputs/attack_plan.jsonl",
    "RL-Remove": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_remove/change_log.jsonl",
    "RL-Add(CAIRAGE)": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_add_cairage/change_log.jsonl",
    "RL-Add(Semantic)": "/home/neha2022/thesis/outputs_thesis_main/dbpedia_output_add_semantic/change_log.jsonl"
}

# Directory to save the output plots
OUTPUT_DIR = "attack_impact_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define color palette
color_map = {
    "Heuristic-Remove": "#4A4D7D",
    "RL-Remove": "#5E8D88",
    "RL-Add(CAIRAGE)": "#7DB66E",
    "RL-Add(Semantic)": "#B0D87E"
}

print(f"Plots will be saved to the '{OUTPUT_DIR}' directory.")

# Data Loading and Processing

def load_jsonl(file_path):
    """Loads data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}. Skipping.")
    return data

def process_attack_data(attack_records):
    """Processes raw attack records to calculate centrality changes."""
    processed_results = []
    for record in attack_records:
        k = record.get('k')
        # Skip records where k is greater than 15
        if k is None or k > 15:
            continue

        before_centralities = record.get('centralities_before_attack', {})
        after_centralities = record.get('centralities_after_attack', {})

        all_nodes = set(before_centralities.keys()) | set(after_centralities.keys())
        if not all_nodes:
            continue

        pagerank_changes = []
        betweenness_changes = []
        eigenvector_changes = []

        for node in all_nodes:
            before = before_centralities.get(node, {})
            after = after_centralities.get(node, {})

            pr_change = abs(after.get('pagerank', 0) - before.get('pagerank', 0))
            bw_change = abs(after.get('betweenness', 0) - before.get('betweenness', 0))
            ev_change = abs(after.get('eigenvector', 0) - before.get('eigenvector', 0))

            pagerank_changes.append(pr_change)
            betweenness_changes.append(bw_change)
            eigenvector_changes.append(ev_change)

        result = {
            'k': k,
            'avg_pagerank_change': np.mean(pagerank_changes) if pagerank_changes else 0,
            'std_pagerank_change': np.std(pagerank_changes) if pagerank_changes else 0,
            'avg_betweenness_change': np.mean(betweenness_changes) if betweenness_changes else 0,
            'std_betweenness_change': np.std(betweenness_changes) if betweenness_changes else 0,
            'avg_eigenvector_change': np.mean(eigenvector_changes) if eigenvector_changes else 0,
            'std_eigenvector_change': np.std(eigenvector_changes) if eigenvector_changes else 0,
            'total_nodes': len(all_nodes)
        }
        processed_results.append(result)

    return processed_results

all_attack_data = []
print("Loading and processing data...")

for strategy_name, file_path in ATTACK_FILES.items():
    print(f"  - Processing {strategy_name} from {file_path}")
    raw_data = load_jsonl(file_path)
    processed_data = process_attack_data(raw_data)

    df = pd.DataFrame(processed_data)
    df['attack_strategy'] = strategy_name
    all_attack_data.append(df)

if not all_attack_data:
    print("No data was loaded. Exiting.")
    exit()

master_df = pd.concat(all_attack_data, ignore_index=True)

# Filter data to include only k values up to 15
master_df = master_df[master_df['k'] <= 15]

# Melt the dataframe for easier plotting
melted_df = master_df.melt(
    id_vars=['k', 'attack_strategy'],
    value_vars=['avg_pagerank_change', 'avg_betweenness_change', 'avg_eigenvector_change'],
    var_name='centrality_metric',
    value_name='average_absolute_change'
)

# Clean up metric names for better plot titles/labels
melted_df['centrality_metric'] = melted_df['centrality_metric'].str.replace('avg_', '').str.replace('_change', '').str.title()

# Set plot style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(sns.color_palette([color_map[name] for name in ATTACK_FILES.keys()]))

print("\nData processing complete. Generating visualizations...")

# Define the k values for the x-axis explicitly. This is crucial for consistent alignment.
k_values_explicit = [3, 5, 7, 9, 11, 13, 15]


# Overall Impact by Attack Type
fig1, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig1.suptitle('Overall Impact of Adversarial Attacks on Graph Centrality', fontsize=16)

for ax, metric in zip(axes, ['Pagerank', 'Betweenness', 'Eigenvector']):
    sns.barplot(
        data=melted_df[melted_df['centrality_metric'] == metric],
        x='attack_strategy',
        y='average_absolute_change',
        ax=ax,
        palette=color_map,
    )
    ax.set_title(f'{metric} Centrality')
    ax.set_xlabel('')
    ax.set_ylabel('Average Absolute Change' if metric == 'Pagerank' else '')
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig1.savefig(os.path.join(OUTPUT_DIR, "rq3_plot1_overall_impact_by_attack_type.png"), dpi=300)
print("  - Overall Impact Plot saved.")

# Impact vs. k
fig2, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig2.suptitle('Impact vs. Number of Modified Triples (k)', fontsize=16)

for ax, metric in zip(axes, ['Pagerank', 'Betweenness', 'Eigenvector']):
    subset = melted_df[melted_df['centrality_metric'] == metric]
    sns.lineplot(
        data=subset,
        x='k',
        y='average_absolute_change',
        hue='attack_strategy',
        style='attack_strategy',
        ax=ax,
        palette=color_map,
        marker='o',
        linewidth=2,
        errorbar=None
    )
    ax.set_title(f'{metric}')
    ax.set_ylabel('Average Absolute Change' if metric == 'Pagerank' else '')
    ax.set_xlabel('k')
    ax.legend(loc='best')

    ax.set_xticks(k_values_explicit)
    ax.set_xticklabels([str(k) for k in k_values_explicit])
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2.savefig(os.path.join(OUTPUT_DIR, "rq3_plot2_impact_by_k_line.png"), dpi=300)
print("Impact vs. k saved.")

# Distribution
fig3, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig3.suptitle('Distribution of Attack Impact Across All Runs', fontsize=16)

for ax, metric in zip(axes, ['Pagerank', 'Betweenness', 'Eigenvector']):
    sns.boxplot(
        data=melted_df[melted_df['centrality_metric'] == metric],
        x='attack_strategy',
        y='average_absolute_change',
        ax=ax,
        palette=color_map,
        hue='attack_strategy',
        legend=False
    )
    ax.set_title(f'{metric} Centrality')
    ax.set_xlabel('' if metric != 'Betweenness' else 'Attack Strategy')
    ax.set_ylabel('Average Absolute Change' if metric == 'Pagerank' else '')
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig3.savefig(os.path.join(OUTPUT_DIR, "rq3_plot3_impact_distribution_boxplot.png"), dpi=300)
print("  - Impact Distribution (Box Plot) saved.")

centrality_metrics_to_plot = ['Pagerank', 'Betweenness', 'Eigenvector']

for metric in centrality_metrics_to_plot:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False)

    fig.suptitle(f'Impact on {metric} Centrality by Attack Strategy and k', fontsize=18)

    metric_df = melted_df[melted_df['centrality_metric'] == metric].copy()
    attack_strategies_ordered = list(ATTACK_FILES.keys())
    max_val_for_current_metric = metric_df['average_absolute_change'].max()

    if max_val_for_current_metric > 0:
        y_limit_for_metric_linear = max_val_for_current_metric * 1.2
        if y_limit_for_metric_linear < 0.001:
            y_limit_for_metric_linear = 0.001
    else:
        y_limit_for_metric_linear = 0.001


    for i, strategy_name in enumerate(attack_strategies_ordered):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        full_k_range_df = pd.DataFrame({'k': k_values_explicit})
        strategy_metric_df_for_plot = pd.merge(
            full_k_range_df,
            metric_df[metric_df['attack_strategy'] == strategy_name],
            on='k',
            how='left'
        ).sort_values(by='k')

        sns.barplot(
            data=strategy_metric_df_for_plot,
            x='k',
            y='average_absolute_change',
            ax=ax,
            color=color_map[strategy_name],
            errorbar=None
        )

        ax.set_title(strategy_name, fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.set_xticks(range(len(k_values_explicit)))
        ax.set_xticklabels([f"k={k}" for k in k_values_explicit])
        ax.tick_params(axis='x', rotation=0)

        if metric == 'Betweenness':
            epsilon = 1e-10
            strategy_metric_df_for_plot['average_absolute_change_for_log_plot'] = strategy_metric_df_for_plot['average_absolute_change'].replace(0, epsilon)
            ax.set_yscale('log')
            min_non_zero_val = strategy_metric_df_for_plot[strategy_metric_df_for_plot['average_absolute_change'] > 0]['average_absolute_change'].min()

            if pd.isna(min_non_zero_val):
                ax.set_ylim(epsilon * 0.5, epsilon * 10)
            else:
                lower_bound = max(epsilon, min_non_zero_val * 0.5)
                upper_bound = strategy_metric_df_for_plot['average_absolute_change'].max() * 2
                ax.set_ylim(lower_bound, upper_bound)
            for container in ax.containers:
                for bar in container:
                    bar_idx = int(bar.get_x() + 0.5)
                    if bar_idx < len(k_values_explicit):
                        current_k = k_values_explicit[bar_idx]
                        original_height_data = strategy_metric_df_for_plot.loc[strategy_metric_df_for_plot['k'] == current_k, 'average_absolute_change'].iloc[0] # Get value based on k

                        if not np.isnan(original_height_data):
                            if original_height_data > 0:
                                # Position slightly above the bar on log scale
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() * 1.05,
                                    f'{original_height_data:.4e}',
                                    ha='center', va='bottom', fontsize=8, color='black'
                                )
                            elif original_height_data == 0:
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2,
                                    ax.get_ylim()[0] * 1.5,
                                    '0.0000',
                                    ha='center', va='bottom', fontsize=8, color='gray'
                                )
            ax.grid(axis='y', which='major', linestyle='-', alpha=0.7)
            ax.grid(axis='y', which='minor', linestyle=':', alpha=0.5)

        else: # For Pagerank and Eigenvector, use linear scale
            ax.set_ylim(0, y_limit_for_metric_linear)
            # Add value labels on top of the bars, but only for non-NaN and non-zero values
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    if not np.isnan(height) and height > 0:
                        label_text = f'{height:.4f}'
                        y_pos = height + (y_limit_for_metric_linear * 0.02)
                        if height < (y_limit_for_metric_linear * 0.05):
                            y_pos = height + (y_limit_for_metric_linear * 0.05)

                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            y_pos,
                            label_text,
                            ha='center', va='bottom', fontsize=8, color='black'
                        )
                    elif not np.isnan(height) and height == 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            y_limit_for_metric_linear * 0.005,
                            '0.0000',
                            ha='center', va='bottom', fontsize=8, color='gray'
                        )
            ax.grid(axis='y', linestyle='-', alpha=0.7)
    if len(attack_strategies_ordered) < 4:
        for j in range(len(attack_strategies_ordered), 4):
            row = j // 2
            col = j % 2
            if row < axes.shape[0] and col < axes.shape[1]:
                fig.delaxes(axes[row][col])
    fig.text(0.5, 0.04, 'k (Number of Modified Triples)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, f'Average Absolute Change in {metric}', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    fig_path = os.path.join(OUTPUT_DIR, f"rq3_barplot_per_strategy_{metric.lower()}.png")
    fig.savefig(fig_path, dpi=300)
    print(f"Saved 2x2 bar plot for {metric} at: {fig_path}")

print("plots generated!!")