import imageProcessor as ip
import numpy as np
import cv2 as cv

def get_image(file=None):
    if(file == None):
        im = ip.thresholdImage()
    else:
        im = ip.thresholdImage(file)
    return np.array(im, copy=False)

if __name__ == "__main__":
    image = get_image('page.jpg')
    cv.imshow("thresholded image", image)
    cv.waitKey()


    