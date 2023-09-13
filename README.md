# Tensor Parallelism in a Toy Model
I train a toy model (3 linear layers with a ReLU between the first and second) to understand tensor parallelism better. I train it on a regression task on a synthetic dataset across 2 GPUs. I MOSTLY use code from the amazing [Megatron repo](https://github.com/NVIDIA/Megatron-LM). I've made the code easy to follow but that's mainly because I stripped away details. Interested readers may refer to page 3 of [my notes](https://www.abhimanyutalwar.com/notes/megatron.html) for the [Megatron paper](https://arxiv.org/abs/1909.08053) to understand how tensor parallelism works in the first two layers of my model. My parallelized Toy Model look like this:

<img src="https://github.com/talwarabhimanyu/tensor_parallel_toy_model/blob/master/assets/toy_tensor_parallel.png" width="500">
Figure: Adapted from Fig. 3(a) of the Megatron-LM Paper. I use ReLU and don't use Dropout.

## Usage
Prepare synthetic data for the regression task:
```
python prepare_data.py 
```

Train the "non-parallel" version of the ToyModel:
```
bash bash_train.sh
```

Train the tensor-parallel version of the ToyModel:
```
bash bash_parallel_train.sh
```

## Notes
You'll see that the training loss curves match for the tensor-parallel version and the "non-parallel" version. There is some work to be done around syncing random number generation on multiple GPUs for tensor-parallel, but to keep it simple, in this repo I have initialized all weight matrices using a simple deterministic scheme.

Here is the train-loss curve for "non-parallel":

<img src="https://github.com/talwarabhimanyu/tensor_parallel_toy_model/blob/master/assets/non_parallel.png" width="300">

For tensor-parallel training:

<img src="https://github.com/talwarabhimanyu/tensor_parallel_toy_model/blob/master/assets/tensor_parallel.png" width="300">


Note that we see the same loss on both GPUs (ranks 0 and 1) because the same regression labels are sent to both devices, and also there is a reduction step which makes sure model output on both devices is the same (refer to page 3 of my [Megatron-LM notes](https://www.abhimanyutalwar.com/notes/megatron.html)).
