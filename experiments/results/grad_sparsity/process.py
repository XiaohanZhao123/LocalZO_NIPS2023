data_path = './cifar10_resent18_surrogate.txt'
import pandas as pd

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
                u_th, beta, num_steps, batch_size = [float(line.strip().split(',')[i].split(':')[1]) for i in
                                                     range(4)]
            # check if line contains layer information
            elif line.startswith("layer_idx"):
                layer_idx = int(line.strip().split(',')[0].split(':')[1])
                sparsity_type, sparsity_value = line.strip().split(',')[1].split(':')
                sparsity_value = float(sparsity_value)
                layer_data.append((layer_idx, sparsity_type, sparsity_value, u_th, beta, num_steps, batch_size))

    # create pandas DataFrame
    df = pd.DataFrame(layer_data, columns=["layer_idx", "sparsity_type", "sparsity_value", "u_th", "beta", "num_steps",
                                           "batch_size"])

    # group by layer_idx and sparsity_type and compute the mean
    df_mean = df.groupby(["layer_idx", "sparsity_type"]).mean().reset_index()

    df_sorted = df_mean.sort_values(by=['sparsity_type', 'layer_idx', ], ascending=[True, True])

    # filter the rows to include only layer_idx values from 1 to 18
    # print the filtered DataFrame
    print(df_sorted)
