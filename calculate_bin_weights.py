import natsort
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm


def build_zenith_bins(n_bins):
    zenith_bins = [0]
    for i in range(1, n_bins):
        delta_theta = np.arccos(np.cos(zenith_bins[i - 1]) - 2 / n_bins) - zenith_bins[i - 1]
        zenith_bins.append(zenith_bins[i - 1] + delta_theta)
    zenith_bins.append(np.pi)
    return np.array(zenith_bins)


def build_weights(df, zenith_bins, factor=1):
    counts, _ = np.histogram(df['zenith'], bins=zenith_bins)
    indices = np.digitize(df['zenith'], zenith_bins[1:])
    return counts, 1 / np.exp((counts[indices] - counts.min()) / (counts.max() - counts.min()) * factor)

n_bins = 30
train_meta_filepathes = natsort.natsorted(
    list(Path('/workspace/data2/train_meta').glob('**/*.parquet')))

# Filter train
train_meta_filepathes = [
    filepath for filepath in train_meta_filepathes
    if int(filepath.stem.split('_')[-1]) <= 655
]

dfs = []
for filepath in tqdm(train_meta_filepathes):
    dfs.append(pd.read_parquet(filepath))
df = pd.concat(dfs)

zenith_bins = build_zenith_bins(n_bins)
counts, df['weight'] =  build_weights(df, zenith_bins, factor=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(np.sin(zenith_bins), np.cos(zenith_bins), '.')
axes[0].set_xlim((-1.1, 1.1))
axes[0].set_ylim((-1.1, 1.1))
axes[0].set_xlabel('x')
axes[0].set_ylabel('z')
axes[0].set_aspect('equal')
axes[0].set_title('Zenith bins')

sns.histplot(df['zenith'], bins=zenith_bins, ax=axes[1])
axes[1].set_title('Counts by zenith bins')

factor = 1
weights = sp.special.softmax(-(counts - counts.min()) / (counts.max() - counts.min()))
g = sns.barplot(x=zenith_bins[:-1], y=weights, errorbar=None, ax=axes[2])
axes[2].set_title('Weights as neg softmax of min-max normed counts')
axes[2].set_xlabel('zenith bin idx')
axes[2].set_ylabel('weight')
axes[2].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.tight_layout()
plt.savefig('weights.png')

df_weights_info = pd.DataFrame({'zenith_bins': zenith_bins[:-1], 'weights': weights})
df_weights_info.to_csv('weights_info.tsv', sep='\t', index=False)
