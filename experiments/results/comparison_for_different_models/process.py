import pandas as pd
import re

file_path = './dvs_guesture_simplenet.txt'
# read the data from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# create an empty list to store the data
data = []

# loop through each line and extract the values
for line in lines:
    match = re.match(r"u_th:(.*), beta:(.*), snn_torch_mean_time:(.*), spconv_mean_time(.*)", line.strip())
    if match:
        u_th = float(match.group(1))
        beta = float(match.group(2))
        snn_torch_mean_time = float(match.group(3))
        spconv_mean_time = float(match.group(4))
        data.append((u_th, beta, snn_torch_mean_time, spconv_mean_time))

# create a pandas dataframe from the data
df = pd.DataFrame(data, columns=['u_th', 'beta', 'LocalZO', 'Surrogate'])

# add a column with speedup, using snntorch time / spconv time
df['Speedup'] = df['LocalZO'] / df['Surrogate']

# rename snn_torch_mean_time to surrogate_mean_time and spconv_mean_time to localzo_mean_time

# group by u_th and beta and take the mean of snn_torch_mean_time and spconv_mean_time
df = df.groupby(['u_th', 'beta']).mean().reset_index()

print(df)
