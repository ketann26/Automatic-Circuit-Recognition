import numpy as np
import cv2
import pandas as pd

import imutils
from imutils.contours import sort_contours
from tensorflow.keras.models import load_model

import utils

def detect_values(src):

	model_path = utils.resource_path("trained_MNIST_model.h5")

	model = load_model(model_path)

	img = src
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	edged = cv2.Canny(blurred, 30, 150)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sort_contours(cnts, method="left-to-right")[0]

	chars = []

	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)

		if (w >= 3 and w <= 50) and (h >= 15 and h <= 50):
			
			roi = gray[y:y + h, x:x + w]			

			thresh = cv2.threshold(roi, 0, 255,
				cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			(tH, tW) = thresh.shape

			if tW > tH:
				thresh = imutils.resize(thresh, width=32)
			# otherwise, resize along the height
			else:
				thresh = imutils.resize(thresh, height=32)

			(tH, tW) = thresh.shape
			dX = int(max(0, 32 - tW) / 2.0)
			dY = int(max(0, 32 - tH) / 2.0)
			# pad the image and force 32x32 dimensions
			padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
				left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
				value=(0, 0, 0))
			padded = cv2.resize(padded, (32, 32))
			# prepare the padded image for classification via our
			# handwriting OCR model
			padded = padded.astype("float32") / 255.0
			padded = np.expand_dims(padded, axis=-1)
			# update our list of characters that will be OCR'd
			chars.append((padded, (x, y, w, h)))

	boxes = [b[1] for b in chars]
	chars = np.array([c[0] for c in chars], dtype="float32")
	# OCR the characters using our handwriting recognition model
	preds = model.predict(chars)
	# define the list of label names
	labelNames = "0123456789"
	labelNames = [l for l in labelNames]

	boxes_val = []

	for (pred, (x, y, w, h)) in zip(preds, boxes):
		# find the index of the label with the largest corresponding
		# probability, then extract the probability and label
		i = np.argmax(pred)
		prob = pred[i]
		label = labelNames[i]

		# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# cv2.putText(img, label, (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

		boxes_val.append([(x,y,w,h), int(label)])

	# print(boxes_val)
	# cv2.imshow('res', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return boxes_val

def combine(boxes_val):

	bvcpy = boxes_val.copy()
	n = 1

	for i in range(len(bvcpy)-2,-1,-1):
		x1 = bvcpy[i][0][0]
		y1 = bvcpy[i][0][1]
		x2 = bvcpy[i+1][0][0]
		y2 = bvcpy[i+1][0][1]
		w1 = bvcpy[i][0][2]
		h1 = bvcpy[i][0][3]
		w2 = bvcpy[i+1][0][2]
		h2 = bvcpy[i+1][0][3]

		dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

		if dist <= 50:
			boxes_val[i][0] = (x1, min(y1,y2), x2-x1+w2, max(y1+h1, y2+h2) - min(y1,y2))
			boxes_val[i][1] = (10**n)*boxes_val[i][1] + boxes_val[i+1][1]

			boxes_val.remove(boxes_val[i+1])
			n = n+1
		
		else:
			n = 1

	return boxes_val

# if __name__ == "__main__":

# 	src = cv2.imread("Sample Images\Circuit 7.jpeg")
# 	src = cv2.resize(src, (640,640))
	
# 	boxes_val = detect_values(src)
# 	combine(boxes_val)
	