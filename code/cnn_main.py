import numpy as np
import cnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

# model train
def model_train(train_data_x, train_data_y, test_data_x, test_data_y, net, optimizer='adam', epoch_num=5, device=torch.device('cpu')):
	net.to(device)
	train_set = TensorDataset(train_data_x, train_data_y)
	trainloader = DataLoader(train_set, batch_size=5, shuffle=True, num_workers=2)
	model_loss = nn.CrossEntropyLoss()
	optimizers = {
				'adam': optim.Adam(net.parameters()),
				'sgd': optim.SGD(net.parameters(), lr=0.02),
				'adagrad': optim.Adagrad(net.parameters(), lr=0.001)
				}
	optimizer = optimizers[optimizer]
	history = {'Epoch':[], 'Test Accuracy':[]}
	for epoch in range(epoch_num):
		running_loss = 0.0
		for i, data in enumerate(trainloader,0):
			inputs, labels = data[0].to(device), data[1].to(device)
			optimizer.zero_grad()

			outputs = net(inputs)
			loss = model_loss(outputs, labels.reshape(-1))
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		this_epoch = epoch + 1
		loss_this_epoch = running_loss / len(train_data_x)
		acc_this_epoch = model_test(test_data_x, test_data_y, net, this_epoch, device)
		print('\nEpoch {}:\n   Loss: {}\n   Accuracy: {}%'.format(this_epoch, loss_this_epoch, acc_this_epoch))
		history['Epoch'].append(this_epoch)
		history['Test Accuracy'].append(acc_this_epoch)
	print("Finished Training")
	return net, history

# model test: can be called directly in model_train
def model_test(test_data_x, test_data_y, net, epoch_num, device):
	test_set = TensorDataset(test_data_x, test_data_y)
	testloader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=2)
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			total += labels.size(0)

			predicted = torch.argmax(outputs, dim=1).reshape(outputs.shape[0],1)
			correct += torch.sum(predicted == labels)
	accuracy = (100*correct/total)
	return accuracy




if __name__ == '__main__':
	# load datasets
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# rescale data
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0
	# model train (model test function can be called directly in model_train)
	device = torch.device('cpu')
	net = U.Net(dropout=False)
	_ , adam = model_train(train_data_x, train_data_y, test_data_x, test_data_y, net, optimizer='adam', epoch_num=100, device=device)
	net = U.Net(dropout=False)
	_ , sgd = model_train(train_data_x, train_data_y, test_data_x, test_data_y, net, optimizer='sgd', epoch_num=100, device=device)
	net = U.Net(dropout=False)
	_ , adagrad = model_train(train_data_x, train_data_y, test_data_x, test_data_y, net, optimizer='adagrad', epoch_num=100, device=device)

	plt.plot(adam['Epoch'], adam['Test Accuracy'], label='Adam')
	plt.plot(sgd['Epoch'], sgd['Test Accuracy'], label='SGD')
	plt.plot(adagrad['Epoch'], adagrad['Test Accuracy'], label='Adagrad')
	plt.axis([0,100,0,100])
	plt.legend()
	plt.xlabel('Epochs')
	plt.ylabel('Test Accuracy (Percent)')
	plt.title('Accuracy over Epochs (CNN without dropout)')

	plt.savefig('history/CNN_nodrop.png')
	plt.close()



	net = U.Net(dropout=True)
	_ , adam = model_train(train_data_x, train_data_y, test_data_x, test_data_y, net, optimizer='adam', epoch_num=100, device=device)
	net = U.Net(dropout=True)
	_ , sgd = model_train(train_data_x, train_data_y, test_data_x, test_data_y, net, optimizer='sgd', epoch_num=100, device=device)
	net = U.Net(dropout=True)
	_ , adagrad = model_train(train_data_x, train_data_y, test_data_x, test_data_y, net, optimizer='adagrad', epoch_num=100, device=device)

	plt.plot(adam['Epoch'], adam['Test Accuracy'], label='Adam')
	plt.plot(sgd['Epoch'], sgd['Test Accuracy'], label='SGD')
	plt.plot(adagrad['Epoch'], adagrad['Test Accuracy'], label='Adagrad')
	plt.axis([0,100,0,100])
	plt.legend()
	plt.xlabel('Epochs')
	plt.ylabel('Test Accuracy (Percent)')
	plt.title('Accuracy over Epochs (CNN with dropout)')
	plt.savefig('history/CNN_drop.png')
	plt.close()
