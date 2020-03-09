import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
import torch.optim as optim
import torch
import pickle

learning_rate = 0.0001
momentum = 0.9
n_epochs = 70
batch_size = 512
device = torch.device("cuda")


''' The main task is to train text_proccesing model on movies revies, that can recognize negative review and positive review 
	
	Training Data represents by itself two text files with positive revies and negative revies, target data is numeric and means rating for each review (from 1 to 10), 
	positive revies - revies with rating >= 7 and negative revies - revies with rating <= 4
'''



def get_batches(data, batch_size):

	''' function that returns batched data '''

	np.random.shuffle(data)
	x = data[:,:-1]
	y = data[:,-1]
	length = len(x)
	x_batched = []
	y_batched = []
	for i in range(int(length/batch_size)):
		x_batch = []
		y_batch = []
		for j in range(batch_size):
			x_batch.append(x[i*batch_size + j][0])
			y_batch.append(y[i*batch_size + j])
		x_batched.append(x_batch)
		y_batched.append(np.array(y_batch).flatten().tolist())	
	return x_batched, y_batched


'''Reding data from text and csv files'''
path = ".../text"
os.chdir(path)

print("Reading revies data...")

f1 = open("pos_revies.txt", "r", encoding="utf8")
all_text_pos = ""

for char in f1:
	all_text_pos += char
f1.close()

f2 = open("neg_revies.txt", "r", encoding="utf8")
all_text_neg = ""

for char in f2:
	all_text_neg += char
f2.close()

f3 = open("test_revies.txt", "r", encoding="utf8")	
all_text_test = ""

for char in f3:
	all_text_test += char	
f3.close()



'''each review in text file are separated from other by ",,,,,?,,,,," separator'''

pos_arr = all_text_pos.replace("<br />", "").split(",,,,,?,,,,,")[:-1]
neg_arr = all_text_neg.replace("<br />", "").split(",,,,,?,,,,,")[:-1]
test_arr = all_text_test.replace("<br />", "").split(",,,,,?,,,,,")[:-1]

pos_len = len(pos_arr)
neg_len = len(neg_arr)



'''Each instance of revies has in itself review index and string text of review separated be "||" separator'''

pos_indices = []
for i in range(pos_len):
	pack = pos_arr[i].split("||")
	pos_indices.append(pack[0])
	pos_arr[i] = pack[1]

neg_indices = []
for i in range(neg_len):
	pack = neg_arr[i].split("||")
	neg_indices.append(pack[0])
	neg_arr[i] = pack[1]	

pos_target = pd.read_csv("pos_rating.csv", sep = ",").values
neg_target = pd.read_csv("neg_rating.csv", sep = ",").values

pos_length = len(pos_arr)
neg_length = len(neg_arr)
train_length = pos_length + neg_length

all_text = pos_arr + neg_arr + test_arr
all_length = len(all_text)



''' As text revies have variable lengths, so each review has been converted to numeric 2000-d vector with help of sklearn.TfidfVectorizer class, 
	which can perform tfidf transormation algorithm on fitted text 
'''
print("Preparing revies data for training...")
vectorizer = TfidfVectorizer(max_features = 2048)
vectorizer.fit(all_text)

vectorized_pos = vectorizer.transform(pos_arr)
shape_pos = np.shape(vectorized_pos)
transformed_pos = np.zeros(shape_pos) + vectorized_pos
transformed_pos = np.hstack((transformed_pos, np.zeros(shape_pos[0]).reshape(shape_pos[0], 1)))
for i in range(shape_pos[0]):
	transformed_pos[i,-1] = 1

vectorized_neg = vectorizer.transform(neg_arr)
shape_neg = np.shape(vectorized_neg)
transformed_neg = np.zeros(shape_neg) + vectorized_neg
transformed_neg = np.hstack((transformed_neg, np.zeros(shape_neg[0]).reshape(shape_neg[0], 1)))
for i in range(shape_neg[0]):
	transformed_neg[i,-1] = 0

vectorized_test = vectorizer.transform(test_arr)
shape_test = np.shape(vectorized_test)
transformed_test = np.zeros(shape_test) + vectorized_test

#transformed via tfidf algorithm review-vectors
train_data = np.vstack((transformed_pos, transformed_neg))



class Net(nn.Module):

	''' simple one-hidden-layer neural network '''

	def __init__(self):
		super(Net, self).__init__()
		

		self.dense_pipe2 =  nn.Sequential(
				nn.Linear(int(shape_pos[1]), 256),
				nn.ReLU(),
				nn.Dropout(p=0.5),
				nn.Linear(256, 1),
				nn.Sigmoid())


	def forward(self, x):	
		x = x.view(-1, int(shape_pos[1]))
		y = self.dense_pipe2(x)
		return y



''' splitting data on training and validation data '''
np.random.shuffle(train_data)
train_data_ = train_data[:24900]
validation_data = train_data[24900:]
train_losses = []
correction_losses = []

net = Net().to(device)
'''defining optimizers '''
loss_objective = nn.BCELoss()
optimizer1 =  optim.Adam(params=net.parameters(), lr=learning_rate, betas=(momentum, 0.999))
optimizer2 =  optim.Adam(params=net.parameters(), lr=learning_rate/5, betas=(momentum-0.2, 0.999))
val_len = len(validation_data)

train_x = train_data[:,:-1]
train_y = train_data[:,-1]


def correction_loss(predictions, labels):
	return torch.mean(-torch.log(predictions + 0.01)*labels)




''' training loop '''

print("Training neural network...")
for i in range(n_epochs):
	batches = get_batches(train_data_, batch_size)
	batch_x_test = batches[0][0]
	for i in range(len(batches[0])):
		batch_x = torch.FloatTensor(np.array(batches[0][i])).to(device)
		batch_y = torch.FloatTensor(np.array(batches[1][i])).to(device)
		optimizer1.zero_grad()
		model_output = net.forward(batch_x)
		loss = loss_objective(model_output.squeeze(1), batch_y)
		train_losses.append(loss.item())
		loss.backward()
		optimizer1.step()




''' validation '''

print("Validation of trained model ...")
batches = get_batches(validation_data, 50)
accuracies_nn = []
mean_errors = []
total = len(validation_data)

for i in range(len(batches[0])):

	val_data_x_batch = torch.FloatTensor(np.array(batches[0][i])).to(device)
	val_data_y_classes = np.array(batches[1][i])
	val_output = net.forward(val_data_x_batch).detach().cpu().numpy()
	val_data_output_classes = np.where(val_output > 0.5, 1, 0)[:,0]
	accuracies_nn.append(sum(val_data_output_classes*val_data_y_classes + (1-val_data_y_classes)*(1-val_data_output_classes)))

nn_accuracy = np.sum(accuracies_nn)/total


''' plotting training losses '''
plt.plot(train_losses)
plt.show()




print(nn_accuracy)




model_file = "revies_model.pth"


test_text = ["that was awful i don't know what to say", 
			"that movie was pretty good",
			"well played actors and interesting scenario",
			"classic instance of its ganre and so realistic, well played actors and good job",
			"not as good as i was expected", 
			"i turned it off after 15 mins",
			"i realy liked it",
			"it was pleasure, i havent seen better movie than this since long time",
			"i will not advise to watch this movie at all",
			"one of the best movies i have ever seen",
			"it was beautiful and i want to forget this movie and watch it again",
			"if you haven't seen it yet you definately have to",
			"too mutch good revies for movie like this",
			"it was worse that i expected"]




transformed_text_sparse = vectorizer.transform(test_text)
t_shape = np.shape(transformed_text_sparse)
transformed_text = np.zeros((t_shape))
transformed_text += transformed_text_sparse

cpu_net = net.to("cpu")

text_tensor = torch.FloatTensor(transformed_text)

stacked = []

for i in range(10):
	stacked.append(cpu_net.forward(text_tensor).detach().cpu().numpy())

stacked = np.mean(np.stack(stacked, axis = 0), axis = 0)

print(stacked)

print("\n\n")

print(" "*2 +"very bad \n" + " "*2 +"good \n" + " "*2 +"nearly good \n" + " "*2 +"good enough \n" + " "*2 +"bad \n" + " "*2 +"very bad \n" + 
		" "*2 +"good \n" + " "*2 +"very good \n" + " "*2 +"very bad \n" + " "*2 +"very good \n" + " "*2 +"very good \n" + " "*2 +"good \n" + " "*2 +"bad \n" + " "*2 +"very bad ")





	






	





	