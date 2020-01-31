'''

This file defines the dataset and data loaders. Please note that there are three common 
ways to define a dataset, and each will be defined in this file. The three main ways are
listed in increasing level of freedom for the programmer:

	1)	Using a PyTorch torchvision dataset (`dataset_torch`)
	2)	Using your own data from a directory containing your data (`dataset_dir`)
	3)	Using/Creating your own data (`Dataset_Custom`)

		Preprocessing of data is available in all three cases.

For a list of readily available image datasets available from PyTorch, please visit
https://pytorch.org/docs/stable/torchvision/datasets.html

IMPORTANT: THE EXAMPLES BELOW ARE NONSENSE FOR OUR NEURAL NETWORK TO CLASSIFY IMAGES.
	WE WILL BE TRYING TO CLASSIFY *RANDOM* DATA AS DIGITS FROM 0-9. HOWEVER, WE WILL
	TEST USING THE MNIST TESTING DATASET. YOU SHOULD BE ABLE TO MODIFY THIS CODE TO
	WORK WITH YOUR DATA AND YOUR NEURAL NETWORK!

'''


import os

from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tforms


'''
The class that is returned in the dataset is not of correct format. It will return an int,
but we want to return a one-hot vector with a 1 in the index of the class. This function
does just that and will be used later in the code.
'''
def int_to_vec(x):
		a = torch.zeros(10)
		a[x] = 1
		return a




'''
TORCHVISION DATASET

Here, we will use a torchvision dataset. Note that you may not need a function declaration
since it may make sense to place this in the main code. However, we will pretend that we
need to preprocess the data. This is only as an example that you can extend to your code.
Here is what the following code will do for a piece of data:

	1)	Create random data of size 3x64x64
	2)	Resize to 32x32
	3)	Randomly crop a 28x28 section to fit our model input
	4)	Take channels 0 and 2 and sum the channels together (1x28x28 output)

torchvision.transforms.Compose collects all the transforms together into one transform.

Note that most transforms in torchvision are for PIL Images, *not* tensors. We will see
how that changes the transforms we need to perform.

Datasets from torchvision return PIL images by default, so typically you will use
torchvision.transforms.ToTensor somewhere in the transforms. If no preprocessing is
required, then torchvision.transforms.ToTensor is enough.
'''
def dataset_torch():
	# Define the transforms
	resize_32x32 = tforms.Resize((32, 32))
	crop_28x28 = tforms.RandomCrop((28, 28))
	sum_channels = tforms.Lambda(lambda x: x[[0],:,:] + x[[2],:,:])

	'''
	The transforms `resize_32x32` and `crop_28x28` work on PIL images only, but
	`extract_channels` works on tensors. We will have to convert to a tensor before
	applying `extract_channels`.

	FYI if no preprocessing is needed, just use torchvision.transforms.ToTensor
	'''
	t = tforms.Compose([resize_32x32, crop_28x28, tforms.ToTensor(), sum_channels])


	# Get the random training dataset (1000 fake images)
	train_data = torchvision.datasets.FakeData(size = 1000,
	                                           image_size = (3, 64, 64),
	                                           num_classes = 10,
	                                           transform = t,
	                                           target_transform = int_to_vec)

	# Get the actual MNIST testing dataset (to get the training dataset, set train = True)
	test_data = torchvision.datasets.MNIST(root = os.path.join(os.getcwd(), 'MNIST_test_data'),
	                                       train = False,
	                                       transform = tforms.ToTensor(),
	                                       target_transform = int_to_vec,
	                                       download = True)

	return train_data, test_data




'''
DIRECTORY DATASET

Here, we assume we have a directory containing our images and labels. The directory
must have a hierarchy if you want to have labels that are not the images themselves:


	my_data_folder/
		my_label_A/
			my_class_A_1.jpg
			my_class_A_2.jpg
			...
		my_label_B/
			my_class_B_1.jpg
			my_class_B_2.jpg
			...
		...


The jpg extension was used as an example. PyTorch will generate the label values
automatically (i.e. `my_label_A` will map to 0 automatically). This means that your
test and validation image foldes should have the exact same hierarchy as your train
folder, but obviously with different datapoints.

Transforms can be applied here as well, but we leave them out here for brevity. See
the `Dataset_Torch` function above as a guide to transforms
'''
def dataset_dir():

	# Get the random training dataset from the `My_Train_Images` directory
	train_data = torchvision.datasets.ImageFolder(root = 'My_Train_Images',
	                                              transform = tforms.ToTensor(),
	                                              target_transform = int_to_vec)

	# Get the actual MNIST testing dataset (to get the training dataset, set train = True)
	test_data = torchvision.datasets.MNIST(root = os.path.join(os.getcwd(), 'MNIST_test_data'),
	                                       train = False,
	                                       transform = tforms.ToTensor(),
	                                       target_transform = int_to_vec,
	                                       download = True)

	return train_data, test_data




'''
CUSTOM DATASET

- Datasets should inherit the torch.utils.data.Dataset class
- All datasets (under the inherited class structure) must have three defined functions:
	`__init__` initializes the class
	`__len__` simply returns the number of data in the dataset
	`__getitem__` defines how you get a single point of data and its label

- Data loaders should inherit the torch.utils.data.DataLoader class

A PyTorch vision dataset skeleton looks like this:


	class My_Dataset(torchvision.datasets.VisionDataset):
		def __init__(self, my_other_arguments, **kwargs):
			super(My_Dataset, self).__init__(root = None, **kwargs)

			# Store any arguments needed here

		def __len__(self):

			return my_num_of_data

		def __getitem__(self, idx):

			# Grab the data and label at index idx

			if self.transform is not None:
				my_data[idx] = self.transform(my_data[idx])
			if self.target_transform is not None:
				my_label[idx] = self.target_transform(my_label[idx])

			return my_data[idx], my_label[idx]


In `__getitem__` for the case of returning images, PIL images should be returned. You
should not return a tensor. This is to keep in line with the other vision datasets
that also return PIL images. It also makes it easier to apply transforms if needed.

Here, we'll pretend that we cannot store all the data from `My_Train_Images` within
our class. Instead, we store a list of file locations that will be read when needed.
Note that this is much slower than reading in all the data or using ImageFolder, but
it gets the job done.
'''
class Dataset_Custom(torchvision.datasets.VisionDataset):
	def __init__(self, **kwargs):
		super(Dataset_Custom, self).__init__(root = None, **kwargs)

		# Collect all the png paths relative to `My_Train_Images`
		self.file_list = []
		for r, d, f in os.walk('My_Train_Images'):
			self.file_list += list(map(lambda x: os.path.relpath(os.path.join(r, x),
			                                                     'My_Train_Images'),
			                           list(filter(lambda x: '.png' in x, f))))

	def __len__(self):

		# Return the total number of images
		return len(self.file_list)

	def __getitem__(self, idx):

		# Get the PIL Image
		im = Image.open(os.path.join('My_Train_Images', self.file_list[idx]))

		# Get the label
		label = int(self.file_list[idx][0])

		# If there are transforms, apply them
		if self.transform is not None:
			im = self.transform(im)
		if self.target_transform is not None:
			label = self.target_transform(label)

		'''
		If you have multiple images per input/output, you can add another transform
		in the `__init__` arguments for your subclass. Then, you can do add more
		code for the additional transforms:

			if self.my_transform_for_image_2 is not None:
				im2 = self.transform(im2)
		'''

		return im, label




'''
CUSTOM DATASET

Here, we just define a function to initialize an instance of our `Dataset_Custom` class
as well as the MNIST testing dataset.
'''
def dataset_custom():

	# Get the random training dataset from the `My_Train_Images` directory
	train_data = Dataset_Custom(transform = tforms.ToTensor(),
	                            target_transform = int_to_vec)

	# Get the actual MNIST testing dataset (to get the training dataset, set train = True)
	test_data = torchvision.datasets.MNIST(root = os.path.join(os.getcwd(), 'MNIST_test_data'),
	                                       train = False,
	                                       transform = tforms.ToTensor(),
	                                       target_transform = int_to_vec,
	                                       download = True)

	return train_data, test_data




'''
DATA LOADERS

Data loaders define how the data should be loaded from you dataset. It includes useful
arguments such as:

	- Batch size 						(batch_size)
	- Random selection					(shuffle)
	- Number of data loading threads	(num_workers)

The argument `num_workers` means that that many subprocesses will be used to load data
outside of the main function. It's useful if you are loading large images/batches.

This function creates data loaders for us. It need not be in a function, and
instead could be defined easily in `train.py`
'''
def create_dataloaders(train_data, test_data):

	# Create a loader for the training data (usually a good idea to shuffle for training)
	train_loader = DataLoader(dataset = train_data,
	                          batch_size = 4,
	                          shuffle = True)

	# Create a loader for the testing data (usually unnecessary to shuffle for testing)
	test_loader = DataLoader(dataset = test_data,
	                         batch_size = 4,
	                         shuffle = False)

	return train_loader, test_loader




'''
This just checks to make sure that the datasets initialize properly.
'''
if __name__ == '__main__':

	# Create a torchvision dataset
	train_data, test_data = dataset_torch()
	train_loader, test_loader = create_dataloaders(train_data, test_data)
	print(train_data[0], test_data[0])

	# Create an ImageFolder dataset
	train_data, test_data = dataset_dir()
	train_loader, test_loader = create_dataloaders(train_data, test_data)
	print(train_data[0], test_data[0])

	# Create a custom dataset
	train_data, test_data = dataset_custom()
	train_loader, test_loader = create_dataloaders(train_data, test_data)
	print(train_data[0], test_data[0])

###