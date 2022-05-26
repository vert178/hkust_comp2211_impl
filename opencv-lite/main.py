from opencv_lite import *
import cv2
import matplotlib.pyplot as plt

desmond = bnw("desmond_bnw.jpeg", save = False)

# threshold_1, desmond_1 = otsu(desmond)
# threshold_2, desmond_2 = cv2.threshold(desmond, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# histogram(desmond)

desmond_1 = equaliseHist(equaliseHist(desmond))
desmond_2 = cv2.equalizeHist(desmond)





plot(desmond_1, desmond_2)
histogram(desmond_1)
histogram(desmond_2)