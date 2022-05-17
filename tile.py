import cv2
import glob
import os
import numpy
import numpy as np
from cv2 import aruco
dir = "."


#images = [cv2.imread(img) for img in glob.glob(pathname)]
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
frame = cv2.imread("/Users/airfly/Downloads/aruco-markers-examples.jpg")
# Load the predefined dictionary
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
for corner in corners:
    x1,x2,x3,x4,y1,y2,y3,y4=corner[0][0][0],corner[0][1][0],corner[0][2][0],corner[0][3][0],\
                            corner[0][0][1],corner[0][1][1],corner[0][2][1],corner[0][3][1]
    centerX = (x1 + x2 + x3 + x4) / 4.00
    centerY = (y1 + y2 + y3 + y4) / 4.00
    center = (int(centerX), int(centerY))
    horz=int(y3)-int(y1)
    vert=int(x3)-int(x1)

    im = frame[center[1]-int(horz):center[1]+int(horz),center[0]-int(vert/2):center[0]+int(vert/2)]
    cnY,cnX,cnZ = im.shape
    show2 =cv2.drawMarker(im, (int(cnX/2),int(cnY/2)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    resized = cv2.resize(show2, (120,100), interpolation=cv2.INTER_AREA)

    cv2.imshow("marker33.png", resized)
    cv2.waitKey(0)  # wait for a keyboard input
    cv2.destroyAllWindows()
#
# img = numpy.zeros((60,60,3), numpy.uint8)
# imagesList =[img01,img02,img01,img02,img02,img02,img01,img02,img01,img02,img02,img02,img01,img02,
#              ]
#
# orgX,orgY,orgZ = imgTest.shape
# print(imgTest.shape)
# x,y =  3, 6
# while len(imagesList)<x*y:
#     imagesList.append(img)
# mat_xy = [imagesList[i*x:(i+1)*x] for i in range(y)]
# #img_tile = numpy.concatenate((img01,img02,img01), axis=1)
# #img_tile = numpy.concatenate((img_tile,img02,img01), axis=0)
#
# print(len(img))
# img_tile = concat_tile(mat_xy)
# cv2.imwrite("tile.jpg",img_tile)
# #cv2.imshow("e",im_tile)
# #cv2.waitKey(0)  # wait for a keyboard input
# #cv2.destroyAllWindows()
#
#
