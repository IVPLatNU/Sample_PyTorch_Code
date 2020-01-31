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

	if model_save_dir is not None and not os.path.exists(model_save_dir):
		os.mkdir(model_save_dir)

	# If you have access to an Nvidia GPU and CUDA, this line will use the GPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Create datasets
	train_data = test_data = []
	if dataset_type == 'torchvision':
		train_data, test_data = dataset.dataset_torch()
	elif dataset_type == 'folder':
		train_data, test_data = dataset.dataset_dir()
	elif dataset_type == 'custom':
		train_data, test_data = dataset.dataset_custom()

	# Create data loader
	train_loader, test_loader = dataset.create_dataloaders(train_data = train_data,
	                                                       test_data = test_data)

	# Initialize the model and load weights if applicable
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

	# HOW TO RUN ONE EPOCH
	def run_model(mode):

		# Enable gradients to be stored only for training
		torch.set_grad_enabled(mode == 'train')

		# Run through the batches
		running_loss = 0
		loader = train_loader if mode == 'train' else test_loader
		for data in loader:

			# Collect the batch information
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			# Clear the gradient for new batch
			if mode == 'train':
				optimizer.zero_grad()

			# Pass the inputs through the network
			outputs = net(inputs)

			# Calculate the loss and update the running loss
			loss = loss_fn(outputs, labels)
			running_loss += loss.item()

			# Update the model weights
			if mode == 'train':
				loss.backward()
				optimizer.step()

		return running_loss

	# Function to save the model
	def save_model(epoch = None):
		model_name = 'model_epoch_{}.pth'.format(epoch) if epoch is not None else 'model_best.pth'
		torch.save(net.state_dict(), os.path.join(model_save_dir, model_name))
		print('\tSaved best model' if epoch is None else '\tCheckpoint saved')


	# TRAINING PROCEDURE
	if 'train' in modes:
		print('Starting training...\n')

		# Initialize statistics
		best_epoch = 0
		best_epoch_loss = 1e9

		# Train the model
		for epoch in range(1, epochs + 1):
			print('Epoch {}:'.format(epoch))

			# Train
			epoch_loss = run_model(mode = 'train')

			print('\tLoss: {:.8f}'.format(epoch_loss))

			# Saving for best model
			if epoch_loss < best_epoch_loss:
				best_epoch_loss = epoch_loss
				best_epoch = epoch
				save_model()

			# Checkpoint saving
			if epoch % save_every == 0 and model_save_dir is not None:
				save_model(epoch)

		# Checkpoint save
		if epoch % save_every != 0:
			save_model(epoch)

		print('\nTrain results: Epoch {} had the best loss of {:.8f}'.format(best_epoch,
		                                                                     best_epoch_loss))

	# TESTING PROCEDURE
	if 'test' in modes:
		if 'train' in modes: print('')
		print('Starting testing...\n')

		# Get test loss
		test_loss = run_model(mode = 'test')

		print('Test results: Loss = {:.8f}'.format(test_loss))


if __name__ == '__main__':
	main(modes = ['train', 'test'],
	     epochs = 10,
	     dataset_type = 'custom',
	     model_save_dir = 'Run_1',
	     save_every = 2)