import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
import torch.optim as optim
import torch
import pickle


MODEL_FILE = "revies_model.pth"
VECTORIZER_FILE = "tfidf_model.pyobj"



class Net(nn.Module):

	''' simple one-hidden-layer neural network '''

	def __init__(self):
		super(Net, self).__init__()
		
		self.dense_pipe2 =  nn.Sequential(
				nn.Linear(2048, 256),
				nn.ReLU(),
				nn.Dropout(p=0.5),
				nn.Linear(256, 1),
				nn.Sigmoid())


	def forward(self, x):	
		x = x.view(-1, 2048)
		y = self.dense_pipe2(x)
		return y

net = Net()


new_text = ["that was awful i don't know what to say", 
			"that movie was pretty good",
			"well played actors and interesting movie",
			"classic instance of its ganre and so realistic, well played actors and good job",
			"not as good as i was expected", 
			"i turned it off after 15 mins",
			"i realy liked it",
			"it was pleasure, i havent seen better movie than this since long time",
			"i will not advise to watch this movie at all",
			"one of the best movies i have ever seen",
			"it was beautiful and i want to forget this movie and watch it again",
			"if you haven't seen it yet you definately have to",
			"too mutch good revies for this movie",
			"it was worse that i expected"]




path = ".../ready_to_use"
os.chdir(path)


with open(VECTORIZER_FILE, "rb") as fv:
	loaded_vectorizer = pickle.load(fv)




transformed_text_sparse_ = loaded_vectorizer.transform(new_text)
t_shape_ = np.shape(transformed_text_sparse_)
transformed_text_ = np.zeros((t_shape_))
transformed_text_ += transformed_text_sparse_


text_tensor_ = torch.FloatTensor(transformed_text_)

print(np.shape(text_tensor_))


net.load_state_dict(torch.load(MODEL_FILE))

stacked = []

for i in range(10):
	stacked.append(net.forward(text_tensor_).detach().cpu().numpy())

stacked = np.stack(stacked, axis = 0)

print(np.mean(stacked, axis = 0))




print("\n\n")

print(" "*8 +"very bad \n" + " "*8 +"good \n" + " "*8 +"nearly good \n" + " "*8 +"good enough \n" + " "*8 +"bad \n" + " "*8 +"very bad \n" + 
		" "*8 +"good \n" + " "*8 +"very good \n" + " "*8 +"very bad \n" + " "*8 +"very good \n" + " "*8 +"very good \n" + " "*8 +"good \n" + " "*8 +"bad \n" + " "*8 +"very bad ")





	






	





