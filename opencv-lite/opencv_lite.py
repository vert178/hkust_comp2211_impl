# An implementation of some of the more "difficult" functions in opencv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# saves an image from colorful to bnw
def bnw(dir, save_name = None, save = True):
    if save_name is None:
        for i in range(len(dir)):
            if dir[i] == ".":
                save_name = dir[:i] + "_bnw" + dir[i:]
                break
    image = mpimg.imread(dir)
    bnw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if save:
        mpimg.imsave(save_name, bnw, cmap = "gray")
    return bnw

# Plots both image at once
def plot(image1, image2):
    r1, c1 = image1.shape
    r2, c2 = image2.shape
    combined = np.zeros((max(r1, r2), c1 + c2), dtype = np.uint8)
    combined[:r1, :c1] = image1
    combined[:r2, c1:] = image2
    plt.figure()
    plt.imshow(combined, cmap = "gray")
    plt.show()
    return combined

# Use otsu binary thresholding
def otsu(image: np.ndarray):
    centroid = np.average(image)
    last = 0
    while centroid != last:
        last = centroid
        whites = image[image > centroid]
        blacks = image[image <= centroid]
        centroid = 0.5 * (np.average(whites) + np.average(blacks))
        print(centroid)
    new_img = np.array(image > centroid, dtype = np.uint8) * 255
    return centroid, new_img

# Plots the histogram and cdf of the image
def histogram(image: np.ndarray):
    hist = cv2.calcHist([image], [0], None, [256], [0,255])
    plt.figure()
    plt.plot(hist)
    plt.show()
    cum_hist = []
    cum = 0
    for i in range(255):
        cum += hist[i][0]
        cum_hist.append([cum])
    plt.plot(cum_hist)
    plt.show()
    return hist

# Use Histogram equalization
def equaliseHist(image:np.ndarray):
    img = np.array(image, dtype = image.dtype)
    hist = cv2.calcHist([image], [0], None, [256], [0,255])
    total_cum = img.shape[0] * img.shape[1]
    cum = 0
    for i in range(255):
        cum += hist[i][0]
        img[image == i] = round(cum / total_cum * 255)
    return img
        
