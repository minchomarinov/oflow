import cv2
import glob
import os
import numpy
import numpy as np

dir = "."


#images = [cv2.imread(img) for img in glob.glob(pathname)]
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

img01 = cv2.imread("fid01.jpg")
img02 = cv2.imread("fid02.jpg")
l =(img01,img02,img02,img02,img01,img02)
x=l
mat_2x5 = [x[i*3:(i+1)*3] for i in range(2)]

#img_tile = numpy.concatenate((img01,img02,img01), axis=1)
#img_tile = numpy.concatenate((img_tile,img02,img01), axis=0)


im_tile = concat_tile(mat_2x5)
#cv2.imwrite("tile.jpg",img_tile)
cv2.imshow("e",im_tile)
cv2.waitKey(0)  # wait for a keyboard input
cv2.destroyAllWindows()

