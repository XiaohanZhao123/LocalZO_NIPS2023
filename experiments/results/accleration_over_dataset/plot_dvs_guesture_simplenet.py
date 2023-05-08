import pandas as pd
import matplotlib.pyplot as plt

# read data
df = pd.read_excel('./dvs_guesture_simplenet.xlsx')

# get unique batch sizes
batch_sizes = df['batch_size'].unique()

# create subplots
fig, axes = plt.subplots(len(batch_sizes), 1, figsize=(5, 16), sharex=True)
fig.subplots_adjust(wspace=0.5, hspace=0.5)

# plot each batch size in a separate subplot
for i, batch_size in enumerate(batch_sizes):
    # select data for current batch size
    batch_df = df[df['batch_size'] == batch_size]

    # plot surrogate and localzo times against time_window
    axes[i].plot(batch_df['time_window'], batch_df['surrogate_mean_time'], '-o', label='Surrogate')
    axes[i].plot(batch_df['time_window'], batch_df['localzo_mean_time'], '-o', label='LocalZO')

    # set subplot title and legend
    axes[i].set_title(f'Batch Size: {batch_size}')
    axes[i].legend()

    # set subplot axis labels
    axes[i].set_ylabel('Time (s)')
    axes[i].set_xlabel('Time Window')

# set overall figure title
fig.suptitle('Execution Time for Different Batch Sizes')

plt.show()
