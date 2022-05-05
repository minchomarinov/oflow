import cv2
import glob
import os
import numpy
dir = "."


#images = [cv2.imread(img) for img in glob.glob(pathname)]
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

img01 = cv2.imread("fid01.jpg")
img02 = cv2.imread("fid02.jpg")

im1_s= img01

#img_tile = numpy.concatenate((img01,img02,img01), axis=1)
#img_tile = numpy.concatenate((img_tile,img02,img01), axis=0)


im_tile = concat_tile([[im1_s, im1_s, im1_s, im1_s],
                       [img02, im1_s, im1_s, im1_s],
                       [im1_s, im1_s, im1_s, im1_s]])
#cv2.imwrite("tile.jpg",img_tile)
cv2.imshow("e",im_tile)
cv2.waitKey(0)  # wait for a keyboard input
cv2.destroyAllWindows()

