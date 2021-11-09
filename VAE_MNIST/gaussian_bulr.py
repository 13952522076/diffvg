import cv2
import numpy

# read image
src = cv2.imread('hr/id1_id4_0.2.png', cv2.IMREAD_UNCHANGED)

# apply guassian blur on src image
dst = cv2.GaussianBlur(src,(5,5),cv2.BORDER_DEFAULT)

# display input and output image
cv2.imwrite("Gaussian Smoothing",numpy.hstack((src, dst)))
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # d
