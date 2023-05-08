import pandas as pd
from matplotlib import pyplot as plt

# hyper parameter for plot
u_th_plot = 0.9
beta_plot = 0.5
data_path = './cifar10_resent18_surrogate.txt'

if __name__ == '__main__':

    u_th = None
    beta = None
    num_steps = None
    batch_size = None
    layer_data = []

    with open(data_path, "r") as f:
        for line in f:
            # check if line contains setup information
            if line.startswith("u_th"):
                u_th, beta, num_steps, batch_size = [line.strip().split(',')[i].split(':')[1] for i in
                                                     range(4)]
                u_th = float(u_th)
                beta = float(beta)
                num_steps = float(num_steps) if num_steps != 'None' else 'None'
                batch_size = float(batch_size)

            # check if line contains layer information
            elif line.startswith("layer_idx"):
                layer_idx = int(line.strip().split(',')[0].split(':')[1])
                sparsity_type, sparsity_value = line.strip().split(',')[1].split(':')
                sparsity_value = float(sparsity_value)
                layer_data.append((layer_idx, sparsity_type, sparsity_value, u_th, beta, num_steps, batch_size))

    df = pd.DataFrame(layer_data, columns=["layer_idx", "sparsity_type", "sparsity_value", "u_th", "beta", "num_steps",
                                           "batch_size"])

    df_mean = df.groupby(["layer_idx", "sparsity_type", 'u_th', 'beta']).mean().reset_index()
    df_sorted = df_mean.sort_values(by=['sparsity_type', 'layer_idx'], ascending=[True, True])

    print('all possible values in u_th', df_sorted['u_th'].unique())
    print('all possible values in beta', df_sorted['beta'].unique())

    df_filtered = df_sorted[(df_sorted['u_th'] == u_th_plot) & (df_sorted['beta'] == beta_plot)]

    # get the unique values of sparsity_type to plot different colors for each
    sparsity_types = df_filtered['sparsity_type'].unique()

    # plot each sparsity_type with a different color
    for sparsity_type in sparsity_types:
        # filter the dataframe to only include rows with the current sparsity_type
        df_sparsity = df_filtered[df_filtered['sparsity_type'] == sparsity_type]

        # plot the layer_idx vs. sparsity_value
        plt.plot(df_sparsity['layer_idx'], df_sparsity['sparsity_value'], label=sparsity_type)

    # set the title and axis labels
    plt.title('$u_{th}$=' + str(u_th_plot) + ', ' '$\\beta=$' + str(beta_plot))
    plt.xlabel("Layer Index (in order of backward pass))")
    plt.ylabel("Sparsity Value")
    plt.ylim(0, 1)

    # show the legend and plot the figure
    plt.legend()
    plt.show()

