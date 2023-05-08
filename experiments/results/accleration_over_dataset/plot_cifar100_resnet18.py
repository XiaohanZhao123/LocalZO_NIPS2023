import pandas as pd
import matplotlib.pyplot as plt

# Load data from file
data = pd.read_excel('./cifar100_resent18.xlsx')

# Create a list of unique batch sizes in the data
batch_sizes = sorted(data['batch_size'].unique())

# Create a new figure with a subplot for each batch size
fig, axs = plt.subplots(len(batch_sizes), 1, figsize=(8, 6 * len(batch_sizes)))

# Loop over each batch size and plot its performance data
for i, batch_size in enumerate(batch_sizes):
    # Get the data for this batch size
    batch_data = data[data['batch_size'] == batch_size]

    # Create a subplot for this batch size
    ax = axs[i]

    # Plot the surrogate time data
    ax.plot(batch_data['num_step'], batch_data['surrogate_mean_time'], label='surrogate', marker='o')

    # Plot the localzo time data
    ax.plot(batch_data['num_step'], batch_data['localzo_mean_time'], label='localzo', marker='o')

    # Set the plot title and axis labels
    ax.set_title(f'Performance for batch size {batch_size}')
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('Time (s)')

    # Add a legend to the plot
    ax.legend()

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()
