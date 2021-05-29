import numpy as np
from tensorflow.keras.datasets import mnist

import matplotlib
matplotlib.use("Agg")

from models import ResNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

def load_mnist_dataset():
	
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
	
	return (data, labels)

if __name__ == "__main__":

	EPOCHS = 50
	INIT_LR = 1e-1
	BS = 128
	
	(data, labels) = load_mnist_dataset()

	data = [cv2.resize(image, (32, 32)) for image in data]
	data = np.array(data, dtype="float32")

	data = np.expand_dims(data, axis=-1)
	data /= 255.0

	le = LabelBinarizer()
	labels = le.fit_transform(labels)
	counts = labels.sum(axis=0)
	# account for skew in the labeled data
	classTotals = labels.sum(axis=0)
	classWeight = {}
	# loop over all classes and calculate the class weight
	for i in range(0, len(classTotals)):
		classWeight[i] = classTotals.max() / classTotals[i]
	# partition the data into training and testing splits using 80% of
	# the data for training and the remaining 20% for testing
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.20, stratify=labels, random_state=42)

	aug = ImageDataGenerator(
		rotation_range=10,
		zoom_range=0.05,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.15,
		horizontal_flip=False,
		fill_mode="nearest")

	opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
		(64, 64, 128, 256), reg=0.0005)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the network
	print("[INFO] training network...")
	H = model.fit(
		aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS,
		class_weight=classWeight,
		verbose=1)

	# define the list of label names
	labelNames = "0123456789"
	labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	labelNames = [l for l in labelNames]
	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=BS)
	print(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=labelNames))

	# save the model to disk
	print("[INFO] serializing network...")
	model.save("trained_MNIST_model", save_format="h5")
	# construct a plot that plots and saves the training history
	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(args["plot"])