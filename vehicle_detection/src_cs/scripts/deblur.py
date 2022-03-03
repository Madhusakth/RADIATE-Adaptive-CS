from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt
import glob

def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")


	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k
	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	# return the output image
	return output


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,help="path to the input image")
args = vars(ap.parse_args())
# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
# construct a sharpening filter
sharpen = np.array(([0, -1, 0],[-1, 5, -1],[0, -1, 0]), dtype="int")
laplacian = np.array(([0, 1, 0],[1, -4, 1],[0, 1, 0]), dtype="int")
# construct the Sobel x-axis kernel
sobelX = np.array(([-1, 0, 1],[-2, 0, 2],[-1, 0, 1]), dtype="int")
# construct the Sobel y-axis kernel
sobelY = np.array(([-1, -2, -1],[0, 0, 0],[1, 2, 1]), dtype="int")

#kernelBank = (("small_blur", smallBlur),("large_blur", largeBlur),("sharpen", sharpen),\
#        ("laplacian", laplacian),("sobel_x", sobelX),("sobel_y", sobelY))
kernelBank = (("small_blur", smallBlur),("sharpen", sharpen))


data_dir = '/home/ms75986/Desktop/Qualcomm/OxRadar/2019-01-10-14-36-48-radar-oxford-10k-partial/radar/radar-recons-cart'

data_path = os.path.join(data_dir,'*npy')
files = glob.glob(data_path)

for images in files:
    gray = np.load(images)
    print(images)
    gray = gray.reshape(gray.shape[0], gray.shape[1])

    #gray = image = cv2.imread(images, cv2.IMREAD_GRAYSCALE)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # loop over the kernels
    for (kernelName, kernel) in kernelBank:
        # apply the kernel to the grayscale image using both
        # our custom `convole` function and OpenCV's `filter2D`
        # function
        print("[INFO] applying {} kernel".format(kernelName))
        convoleOutput = convolve(gray, kernel)
        opencvOutput = cv2.filter2D(gray, -1, kernel)
        cart_img = opencvOutput
        print(cart_img.shape)
        #plt.imshow(cart_img.reshape(cart_img.shape[0], cart_img.shape[1]))
        if kernelName == 'sharpen':
            image_name = data_dir[:-18] + '/radar-cart-deblur' + images[-21:-4] + '.png' #+ '_cart.png'
            #image_name = data_dir + '/radar-cart-deblur' + images[-21:] #+ '_cart.png'
            #plt.savefig(image_name)
            #plt.clf()
	    # show the output images
            #cv2.imwrite(image_name, cart_img)
            cv2.imshow("original", gray)
            #cv2.imshow("{} - convole".format(kernelName), convoleOutput)
            cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()




