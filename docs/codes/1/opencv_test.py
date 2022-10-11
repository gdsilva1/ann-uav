import cv2 as cv


# LOAD THE IMAGE
# The second parameter is set to 0 to load the image as grayscale
img = cv.imread('railway3.png', 0)

# CONVERTING TO BINARY
# img_bi is the actual image
# First parameter is the image path
# Second parameter is the limit where the method should decide if a pixel gonna
# be black or white
# Third parameter is the maximum value of a calor (remeber RGB scale)
# Fourth parameter set the threshhold type
thresh, img_bi = cv.threshold(img, 120, 255, cv.THRESH_BINARY)

# SHOWING THE IMAGE
cv.imshow('Display Windows', img_bi)
# cv.imshow('Display Windows', img)
k = cv.waitKey(0)