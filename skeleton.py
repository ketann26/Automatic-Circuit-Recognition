import numpy as np
import cv2
import imutils
from skimage.morphology import skeletonize as skt

def skeletonize(img):

	img = img.copy()
	np_img = np.array(img)
	np_img[np_img > 0] = 1

	skel = skt(np_img)
	
	skel = skel.astype(np.uint8)

	skel[skel == 1] = 255

	return skel

def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])

    filtered = cv2.filter2D(skel,-1,kernel)

    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 255
    return out

if __name__ == "__main__":

	src = cv2.imread(r"F:\Python\Circuit Solver\Circuit 1.jpeg")
	src = imutils.resize(src,width=640)
	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

	img = cv2.GaussianBlur(gray,(9,9),0)
	th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
			cv2.THRESH_BINARY_INV,5,2)

	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	
	dilated = cv2.dilate(th, element, iterations=3)
	bw = skeletonize(dilated)
	

	cv2.imshow("res",bw)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
