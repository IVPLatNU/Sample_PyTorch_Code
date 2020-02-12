# PyTorch Tutorial
This is sample code for how to use [PyTorch](https://pytorch.org) for your projects.

## Problem
In this example, we want to try to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data. However, we have cleverly made some important changes to prevent you from downloading this code and submitting it as your project without some major changes. Don't worry, you'll still learn how to use PyTorch through this odd example.

### The Data
The training set we use is *not* the MNIST dataset. Rather, we use randomly generated image data as the training set. When we test our network on the actual MNIST testing set, we will get garbage results that amount to the neural network being as good as random guessing.

The input data consists of 1x28x28 images. That is, a single channel image (intensity) of size 28x28 (rows x columns). Our labels consist of integers corresponding to the class of the image. In our code, we will show how to easily turn this into a one-hot encoded vector.

We have three different training datasets (consisting of randomly generated images) in `dataset.py` to choose from, each serving as an example for you to use in your own code:

1. A `torchvision` dataset: If you are just looking to use one of the datasets readily available from PyTorch, then this is the example for you. A list of these datasets are available [here](https://pytorch.org/docs/stable/torchvision/datasets.html).
2. An `ImageFolder` dataset: If you have your own data that's nicely organized in its own directory, then this is the example for you. Note that this class can only read in one image as your input. It likely wouldn't be as good for, say, a network to compute optical flow from multiple images.
3. A `VisionDataset` dataset: If the other above two dataset types don't suit your needs, then this is the example for you. This example dataset shows how you can get the most control over your dataset.

The testing dataset is the MNIST `torchvision` dataset. Within the example code, we explain how you can get the training data as well.

## Solution

We will train a convolutional neural network. It consists of the following layers:

- 2D Convolutions
- Fully connected/Dense/Linear
- Batch normalization
- ReLU activation

The output will be a one-hot encoded vector, i.e. a 1-dimensional vector of size 10 where the elements are all zero except at the index corresponding to the correct class, which will be one.

Our loss function will simply be the mean squared error.

---

<p align="center"><img src="https://github.com/IVPLatNU/Sample_PyTorch_Code/blob/master/misc/network.png" width="60%"></p>

---

## The Code
The main function is within `run_model.py`. To do a basic run, just type in the command line: 
```
python run_model.py
```

### `run_model.py`
This file is an example of how to train and test your own model.

There are a number of parameters that you can change in here. The arguments for `main` are:

- `modes`: A list or string containing a subset of `['train', 'test']`
- `epochs`: Number of training epochs
- `dataset_type`: A string chosen from `['torchvision', 'folder', 'custom']`. See `dataset.py` for more details.
- `model_load_path`: The load path of a saved model
- `model_save_dir`: The save directory for models saved during training
- `save_every`: Number of epochs to train before checkpoint saving

### `model.py`
This file defines the neural network and the loss function. Note that there are some useful functions that you may want to use to make designing your own network much easier.

### `dataset.py`
This file defines the dataset and data loaders. As was mentioned earlier, three different ways to define your dataset are shown in this file as an example for you.

---

Please direct any comments/questions about this example code to <ivpl@u.northwestern.edu> or raise an issue in this repo.