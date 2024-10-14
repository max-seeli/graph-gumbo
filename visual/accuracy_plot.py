import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Define color palette
palette = sns.color_palette("muted")

dataset_tags = {
    "IMDB": ["IMDB-BINARY 0 3", "IMDB-BINARY 1 3", "IMDB-BINARY 2 3"],
    "REDDIT": ["REDDIT-BINARY 0 0", "REDDIT-BINARY 1 0", "REDDIT-BINARY 2 1"],
    "SYNTHETIC (no attr)": ["SYNTHETIC (no attr) 0 3", "SYNTHETIC (no attr) 1 2", "SYNTHETIC (no attr) 2 3"],
    "MUTAG": ["MUTAG 0 0", "MUTAG 1 1", "MUTAG 2 3"],
    "ENZYMES": ["ENZYMES 0 1", "ENZYMES 1 1", "ENZYMES 2 1"],
}

names = ["deg", "npbcc-cart", "npbcc-mod"]

def get_tagged_data(tag):
    runs = wandb.Api().runs("max-seeli/neural-graph-gumbo", filters={"tags": {"$in": [tag]}})
    data = []
    for run in runs:
        history = run.history(keys=["train_accuracy"])
        data.append(history["train_accuracy"].values)
    return data



def draw_graph(dataset_name, ax):
    # Get the data for each tag
    data = [get_tagged_data(tag) for tag in dataset_tags[dataset_name]]



    for i, tag_data in enumerate(data):
        tag_data = [np.array(d) for d in tag_data if len(d) > 0]  # Convert lists to numpy arrays and filter empty lists
        if len(tag_data) == 0:
            continue
    
        # Pad sequences to the same length
        max_len = max(len(d) for d in tag_data)
        padded_data = np.array([np.pad(d, (0, max_len - len(d)), mode='constant', constant_values=np.nan) for d in tag_data])
    
        # Compute mean and standard deviation while ignoring NaNs
        mean = np.nanmean(padded_data, axis=0)
        std = np.nanstd(padded_data, axis=0)
    
        # Plot mean and shaded standard deviation
        ax.plot(mean, label=names[i], color=palette[i % len(palette)], linewidth=4)
        ax.fill_between(range(len(mean)), mean - std, mean + std, color=palette[i % len(palette)], alpha=0.3)

#plt.ylim(0.6, 1)
    ax.set_ylim(0.4, 1)
    ax.set_xlabel("Epoch", fontsize=24)
    ax.set_ylabel("Train Accuracy", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)

    #ax.legend(fontsize=30)
    
    ax.set_title(dataset_name, fontsize=35)
    
    return ax

fig, axs = plt.subplots(2, 3, figsize=(30, 20))

for i, dataset_name in enumerate(dataset_tags.keys()):
    print(i // 3, i % 3)
    draw_graph(dataset_name, axs[i // 3, i % 3])
    
    
axs[1, 2].axis('off')
lines, labels = axs[0, 0].get_legend_handles_labels()
axs[1, 2].legend(lines, labels, loc='center', fontsize=54)    

    
fig.tight_layout(pad=4.0, h_pad=9.0)

plt.savefig("visual/plots/train_accuracy.pdf")
plt.show()
