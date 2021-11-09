import numpy as np
import cv2


image = cv2.imread("demo5.png")#读入图像
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像
# cv2.imshow("Image",image)#显示图像
# cv2.waitKey()

#Laplacian
edge = cv2.Laplacian(image,cv2.CV_64F)
edge = np.uint8(np.absolute(edge))

#
edge = cv2.Canny(image,30, 50)
cv2.imshow("edge",edge)
cv2.waitKey()
print(edge)
print(edge.shape)
