'''

This file defines how to train and test the neural network.

The main function takes the following arguments:

	- modes: A list containing a subset of ['train', 'test']
	- epochs: Number of training epochs
	- dataset_type: A string from ['torchvision', 'folder', 'custom'].
					See dataset.py for more details.
	- model_load_path: Load path of a saved model
	- model_save_dir: Save directory for models saved during training
	- save_every: Number of epochs to train before checkpoint saving

This code can be used as a skeleton for your own code.

'''


import os

import model
import dataset

import torch

def main(modes,
         epochs = 1,
         dataset_type = 'torchvision',
         model_load_path = None,
         model_save_dir = None,
         save_every = 100):

	'''
	This beginning section is mainly for initialization of everything. Once everything
	is initialized, we then define how to use the network.
	'''

	# Create a save directory if it doesn't already exist
	if model_save_dir is not None and not os.path.exists(model_save_dir):
		os.mkdir(model_save_dir)

	'''
	If you have access to an Nvidia GPU and CUDA, this line will use the GPU.  It will
	check automatically for you. For data that you want to send to the GPU, use the `to`
	method, callable from the data. When we initialize the network for example, we use
	the `to` function.

	Common items to send to the GPU are:

		- The network
		- Inputs
		- Outputs
		- Labels
		- Loss function

	You do *not* send the optimizer to the GPU
	'''
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Initialize datasets
	train_data = test_data = None
	if dataset_type == 'torchvision': train_data, test_data = dataset.dataset_torch()
	elif dataset_type == 'folder': train_data, test_data = dataset.dataset_dir()
	elif dataset_type == 'custom': train_data, test_data = dataset.dataset_custom()

	# Initialize data loaders
	train_loader, test_loader = dataset.create_dataloaders(train_data = train_data,
	                                                       test_data = test_data)

	'''
	Initialize the model and load weights if applicable

	To load a trained model, we load the `state_dict` of the model, which contains
	information about the weights themselves as well as where the weights go within
	the network.
	'''
	net = model.Example_Network().to(device)
	if model_load_path is not None:
		try:
			net.load_state_dict(torch.load(model_load_path))
		except:
			net.load_state_dict(torch.load(model_load_path,
			                               map_location = device))

	# Initialize the loss function
	loss_fn = model.Loss_Function().to(device)

	'''
	Initialize the optimizer:

		lr:		Learning rate
		betas:	Adam momentum terms

	For other optimizers, visit https://pytorch.org/docs/stable/optim.html
	'''
	optimizer = None
	if 'train' in modes:
		optimizer = torch.optim.Adam(params = net.parameters(),
		                             lr = 1e-4,
		                             betas = (0.9, 0.999))

	'''
	HOW TO RUN ONE EPOCH

	This function tells the network how to run an epoch based on the mode.
	If the mode is 'train', then it will train the network. If the mode is
	'test', then it will provide additional statistics for misclassification.
	Regardless, a lot of the train/val/test code (val not done here) is similar,
	so it makes sense to join them into one function.
	'''
	def run_epoch(mode):

		# Initialize statistics
		running_loss = 0
		misclass = 0 if mode == 'test' else None

		# Get the right data loader
		loader = train_loader if mode == 'train' else test_loader

		'''
		Run through each batch

		To get the data within a batch, all you need to do is iterate through
		the loader. It will collect a batch automatically based on the parameters
		used to initialize it.
		'''
		for data in loader:

			'''
			Clear the gradient for new batch, i.e. don't accumulate the gradient
			from the previous batch. Instead, reset the gradient.
			'''
			if mode == 'train': optimizer.zero_grad()

			'''
			Collect the data from the batch

			As you iterate, `data` will be a tuple containing the inputs and labels if
			you used a torchvision dataset including ImageFolder. For custom datasets,
			the `__getitem__` method determines the structure of the iterable. If we
			see the `__getitem__` method of Dataset_Custom within dataset.py, we give
			it the same return variables as a predefined torchvision dataset.
			'''
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			'''
			Pass the inputs through the network

			Our network only takes in one set of image data as defined in
			Example_Network's `forward` function. You can modify it to have more
			inputs should your netowrk require it.
			'''
			outputs = net(inputs)

			'''
			Calculate the loss and update the running loss

			The `item` function of a tensor gives a Python int, float, etc.
			instead of a PyTorch tensor.
			'''
			loss = loss_fn(outputs, labels)
			running_loss += loss.item()

			'''
			Update the model weights for training only

			The `backward` function backpropogates to find the gradient of the
			loss function. The `step` function in the optimizer then carries out
			the weight update step. This is obviously only needed for training.
			'''
			if mode == 'train':
				loss.backward()
				optimizer.step()

			# Count the number of misclassifications for testing only
			elif mode == 'test':

				'''
				Extract the integer class labels

				Here, our outputs and labels are of size (N,C) where N is the
				size of the batch, and C is the number of classes (10). They are
				one-hot vectors, so we want to compare whether or not the argmax
				of each row matches between `outputs` and `labels`.
				'''
				outputs_class = torch.argmax(outputs, dim = 1)
				labels_class = torch.argmax(labels, dim = 1)

				# Accumulate misclassifications
				misclass += (outputs_class != labels_class).sum().item()

		return running_loss, misclass


	'''
	Function to save the model

	To save the model, you don't save the network, but rather its `state_dict` which
	contains the weights of the parameters among other things. Typically we use a
	.pth or .pt extension
	'''
	def save_model(epoch = None):

		# Get model name
		model_name = 'model_epoch_{}'.format(epoch) if epoch is not None else 'model_best'

		# Save the model
		torch.save(net.state_dict(), os.path.join(model_save_dir, model_name + '.pth'))

		print('\tSaved best model' if epoch is None else '\tCheckpoint saved')


	'''
	TRAINING PROCEDURE

	Here, we define how to train the model. There are some preparations that need to
	be done before calling the `run_epoch` function in a loop. Those steps are
	described in detail below.
	'''
	if 'train' in modes:
		print('Starting training...\n')

		'''
		Enable gradients to be stored

		The default setting is that gradients are stored, so this line isn't necessary.
		But why risk it? During testing, we turn this off since the gradients need not
		be calculated.
		'''
		torch.set_grad_enabled(True)

		'''
		Allow model training

		This tells our network that we intend to train it. This line of code is mainly
		for batch normalization and dropout layers. It tells the network that we should
		be using these layers for training.
		'''
		net.train()

		# Initialize statistics
		best_epoch = 0
		best_epoch_loss = 1e9

		'''
		Train the model

		Here, we run the training data through our network for however many epochs
		we defined. Training statistics are printed to show that our model is actually
		training, and can also be used to determine when to stop training. Early
		stopping of training can easily be programmed if, for example, our loss
		decreases by a small margin a certain number of times. This is not coded here,
		and is left for you should you want an early stopping criterion of any sort.
		'''
		for epoch in range(1, epochs + 1):
			print('Epoch {}:'.format(epoch))

			# Train for one epoch
			epoch_loss, _ = run_epoch(mode = 'train')

			print('\tLoss = {:.8f}'.format(epoch_loss))

			# Save the weights if the new model produces a lower loss
			if epoch_loss < best_epoch_loss:
				best_epoch_loss = epoch_loss
				best_epoch = epoch
				save_model()

			# Checkpoint save
			if epoch % save_every == 0 and model_save_dir is not None:
				save_model(epoch)

		# Save the last set of weights
		if epoch % save_every != 0:
			save_model(epoch)

		print('\nTrain results: Epoch {} had the best loss of {:.8f}'.format(best_epoch,
		                                                                     best_epoch_loss))

	'''
	TESTING PROCEDURE

	Like the testing procedure, there are some items to do before running `run_epoch`
	over the testing data. Each step will be described in detail.
	'''
	if 'test' in modes:
		if 'train' in modes: print('')
		print('Starting testing...\n')

		'''
		Disable gradients from being stored

		Since we are testing, we do not need to store the gradients. Gradients are
		only needed when we train so the optimizer can update the network weights.
		'''
		torch.set_grad_enabled(False)

		'''
		Ignore batch norm and dropout layers (inference mode)

		Here, we tell the network to ignore certain layers. For example, we do not
		want to apply dropout when we test, since that is a training-specific layer.
		The `eval` function does just that for us without having to define a new
		testing model without dropout and batch normalization.
		'''
		net.eval()

		# Test the network
		test_loss, misclassifications = run_epoch(mode = 'test')

		# Calculate the network's accuracy
		accuracy = 100 * (1 - misclassifications / len(test_data))

		print('Testing results:')
		print('\tLoss = {:.8f}'.format(test_loss))
		print('\tMisclassifications = {}/{}'.format(misclassifications, len(test_data)))
		print('\tAccuracy = {:.4f}%'.format(accuracy))


if __name__ == '__main__':
	main(modes = ['train', 'test'],
	     epochs = 10,
	     dataset_type = 'custom',
	     model_save_dir = 'Run_1',
	     save_every = 2)