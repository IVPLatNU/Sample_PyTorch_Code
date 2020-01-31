'''

This file defines the neural network and the loss function.

- All neural networks must inherit the nn.Module class
- All networks must have two defined functions:
	`__init__` initializes the class
	`forward` defines the forward pass of data through the model

A PyTorch neural network skeleton looks like this:


	class My_Network(nn.Module):
		def __init__(self, my_other_arguments):
			super(My_Network, self).__init__()

			# Network *layers* go here

		def forward(self, my_batch_data):

			# Network *structure/order* goes here

			return network_output


PyTorch's torch.nn module has many prebuilt functions that may be useful:
	Fully connected layers	(nn.Linear)
	Convolutional layers	(nn.Conv2d)
	Pooling layers			(nn.MaxPool2d, nn.AvgPool2d)
	Dropout layers			(nn.Dropout2d)
	Activation functions	(nn.ReLU, nn.Sigmoid)
	Loss functions			(nn.MSELoss, nn.CrossEntropyLoss)

Please reference https://pytorch.org/docs/stable/nn.html for all things neural net

'''


import torch
import torch.nn as nn


'''
This is the main network class that defines the layers and structure.
The loss function is *not* a part of the main network
'''
class Example_Network(nn.Module):
	def __init__(self, n_blocks = 4, n_channels = 16):
		super(Example_Network, self).__init__()

		self.conv_in = nn.Conv2d(in_channels = 1,
		                         out_channels = n_channels,
		                         kernel_size = 3,
		                         stride = 1,
		                         padding = 1,
		                         bias = False)

		self.inner_convs = self.make_layers(block = Inner_Conv_Block,
											n_blocks = n_blocks)

		self.conv_out = nn.Conv2d(in_channels = n_channels,
		                          out_channels = 1,
		                          kernel_size = 3,
		                          stride = 1,
		                          padding = 1,
		                          bias = False)

		'''
		Inner_Conv_Block decreases the spatial resolution of the original image (28x28).
		No padding is used for a kernel of size 3x3, so each dimension decreases by 2
		for each of the `n_block` convolutions. The variable `output_size` calculates
		the output size based on `n_blocks`.
		'''
		self.output_size = 28 - (2 * n_blocks)

		self.inner_fc1 = Inner_Linear_Block(in_features = self.output_size * self.output_size,
		                                    out_features = 20)

		self.inner_fc2 = Inner_Linear_Block(in_features = 20,
		                                    out_features = 10)		

		self.fc_out = nn.Linear(in_features = 10,
		                        out_features = 10)

		self.relu = nn.ReLU()

	def forward(self, input):
		output = self.relu(self.conv_in(input))
		output = self.inner_convs(output)
		output = self.relu(self.conv_out(output))

		'''
		When you switch from convolutions to fully connected layers, PyTorch requires
		that the shape of the data be adjusted (flattened).
		'''
		output = output.view(-1, self.output_size * self.output_size)

		output = self.inner_fc1(output)
		output = self.inner_fc2(output)
		output = self.fc_out(output)

		return output


	'''
	This is a great function that creates many identical layers sequentially.
	Here, we will use it to create many Inner_Conv_Block layers in our network.
	'''
	def make_layers(self, block, n_blocks):
		layers = [block()] * n_blocks
		return nn.Sequential(*layers)


'''
Inner_Conv_Block contains three layers:
	1)	Convolution
	2)	Batch Normalization
	3)	ReLU Activation
'''
class Inner_Conv_Block(nn.Module):
	def __init__(self, n_channels = 16):
		super(Inner_Conv_Block, self).__init__()

		self.conv = nn.Conv2d(in_channels = n_channels,
		                      out_channels = n_channels,
		                      kernel_size = 3,
		                      stride = 1,
		                      padding = 0,
		                      bias = False)

		self.batch_norm = nn.BatchNorm2d(num_features = n_channels)

		self.relu = nn.ReLU()

	def forward(self, input):
		return self.relu(self.batch_norm(self.conv(input)))


'''
Inner_Conv_Block contains three layers:
	1)	Fully Connected
	2)	Batch Normalization
	3)	ReLU Activation
'''
class Inner_Linear_Block(nn.Module):
	def __init__(self, in_features, out_features):
		super(Inner_Linear_Block, self).__init__()

		self.lin = nn.Linear(in_features = in_features,
		                     out_features = out_features)

		#self.batch_norm = nn.BatchNorm1d(num_features = out_features)

		self.relu = nn.ReLU()

	def forward(self, input):
		return self.relu(self.lin(input))


'''
You don't need a class for the loss function if it's predefined in torch.nn.
This loss function is here as an example for how to do a custom loss function.
Here we use a sigmoid function with the mean squared error.
'''
class Loss_Function(nn.Module):
	def __init__(self):
		super(Loss_Function, self).__init__()

		self.sig = nn.Sigmoid()
		self.mse = nn.MSELoss()

	def forward(self, prediction, target):
		return self.mse(self.sig(prediction), target)



'''
This just checks to make sure that the network works/initializes properly.
'''
if __name__ == '__main__':
	
	# Set the seed for reproducibility
	torch.manual_seed(0)

	# Example image and class
	test_im = torch.zeros((1,1,28,28))
	test_class = torch.tensor([[0,0,0,1,0,0,0,0,0,0]],
	                          dtype = torch.float)

	# Initialize the network and loss function
	net = Example_Network()
	loss = Loss_Function()

	# Forward pass without keeping track of the gradient
	with torch.no_grad():
		net_output = net(test_im)
		net_loss = loss(net_output, test_class)

	print(net_output)
	print(net_loss)

###