import tensorflow
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import random
import json
import tflearn
import pickle


with open("intents.json") as file:
	data = json.load(file)

words = []
train_x = []
train_y = []
labels = []
try:

	with open("data.pickle","rb") as f:
		labels,words,training,output = pickle.load(f)
except:
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			wrds = [stemmer.stem(w.lower()) for w in wrds]
			words.extend(wrds)

			train_x.append(pattern)
			train_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	words = list(set(words))

	training = []
	output = []

	for i,doc in enumerate(train_x):
		bag = []
		wrds = nltk.word_tokenize(doc)
		wrds = [stemmer.stem(w.lower()) for w in wrds]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		out = [0 for _ in range(len(labels))]
		out[labels.index(train_y[i])] = 1

		training.append(bag)
		output.append(out)

	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.picke","wb") as f:
		pickle.dump((labels,words,training,output),f)

tensorflow.compat.v1.get_default_graph()

net = tflearn.input_data(shape=(None,len(training[0])))
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	X #this line here creates an error so that control goes to EXCEPT part
	model.load("tflearn2.model")
except:
	model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
	model.save("tflearn2.model")



def pre_process(inp):
	bag = []
	inp = nltk.word_tokenize(inp)
	inp = [stemmer.stem(i.lower()) for i in inp]

	for w in words:
		if w in inp:
			bag.append(1)
		else:
			bag.append(0)

	return numpy.array(bag)

def chat():
	print("Bot : Hi there! Type quit to exit")
	while(True):
		inp = input("You : ")
		if(inp == "quit"):
			break

		res = model.predict([pre_process(inp)])
		max_i = numpy.argmax(res)
		responses = data["intents"][max_i]["responses"]

		response = random.choice(responses)
		print("Bot : ",response)

chat()