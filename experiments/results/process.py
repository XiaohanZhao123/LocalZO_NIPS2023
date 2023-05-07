import pandas as pd
from io import StringIO

path = './cifar100_resent18_snntorch_time.txt'


if __name__ == '__main__':
    # read the results from a text file into a pandas DataFrame
    with open(path, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            values = line.strip().split(', ')
            d = {'batch_size': values[0].split(':')[1],
                 'num_step': values[1].split(':')[1] if values[1].split(':')[1] != 'None' else -1,
                 'snn_torch_mean_time': float(values[2].split(':')[1]),
                 'spconv_mean_time': float(values[3].split('mean_time')[1])}
            data.append(d)

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Compute speedup
        df['speedup'] = df['snn_torch_mean_time'] / df['spconv_mean_time']

        # Display the DataFrame
        print(df)
        # Display the DataFrame
        df.to_excel('./cifar100_resent18_snntorch_time.xlsx')


