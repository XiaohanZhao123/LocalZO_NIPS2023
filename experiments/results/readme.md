# Experimental Results for Local ZO with Convolutions

## 1. Files

* *acceleration_over_dataset*: Compare the speed of surrogate and localzo with different setup of **batch size** and **num_steps (in dvs datasets like dvs-cifar10, time windos instead)**. The time_window is for length of period to sum up all the events in row dataset.
* *grad_sparsity* of gradient among different layer in the order of backward propagation.
* *acceleration_over_models*: Compare the speed of surrogate and localzo with different setup of model parameters, i.e. **$\beta$** and **$u_{th}$**

## 2. How to use the data

### 2.1 acc over dataset

The data has been processed and saved into .xlsx already.
Demos for how to read and plot the data are provided with name polt_{file_name}.py

### 2.2 grad sparsity

The data is organized in the following way:

A group of data begins with like u_th:0.0,beta:0.0,num_steps:50,batch_size:128, then we have layer index and sparsity of input and output of the gradient. The data for same layer idx may occur multiple times. Also, a demo for how to read and taking average of these data and organized them in pandas is given in process.py. Some of its parameters are explained below:

u_th_plot : the $u_{th}$ of the model to be ploted
beta_plot : the $\beta$ of the model to be plot
data_path : the data path for reading

I personally recommend using smaller value for $u_{th}$ as the curve looks better and we can still preserve sparse gradient even with $u_{th}=0.1,0.2$. We can show specific layers.

Setting for dataset is given below:
cifar10: batch size 32, num steps 25

cifar100: batch size 32, num steps 25

dvs_guesture: batch size 16, time window 60000

dvs_cifar10: batch size 8, time window 90000

### 2.3 acc over models

The test for speed up over surrogate with $u_{th} \in [0,1]$ and $\beta \in [0,1]$ with step size 0.1. Name pattern for file is {dataset}_{model}__{algorithm}
The data is recoded as u_th:{value}, beta:{value}, snn_torch_mean_time:{value}, spconv_mean_time{value},
A demo for how to read the data is given. Some combination may be missing due to OOM failure
