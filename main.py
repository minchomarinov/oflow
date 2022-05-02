import cv2
import numpy
import numpy.distutils.armccompiler

TEMP_FOLDER="Data/"
TEMP_FILE1="car.01.jpg"
TEMP_FILE2="car.02.jpg"

def read_image(thefile):

    img = cv2.imread(thefile)
    cv2.imshow('image', img)


def oflow():
    # Parameters for lucas kanade optical flow
    img1 = cv2.imread(TEMP_FILE1)
    img2 = cv2.imread(TEMP_FILE2)
    grayimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(150, 150),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(grayimg1,
                                           grayimg2,
                                           p0, None,
                                           **lk_params)


def main():
    read_image(TEMP_FILE1)

#def draw_marker(img, position):
#    cv2.drawMarker(img, (position[1], position[0]), (0,255,0))


if __name__ == '__main__':
    #img = cv2.imread('D:/image-1.png', cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(TEMP_FILE1)
    img2 = cv2.imread(TEMP_FILE2)
    grayimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    x = 458
    y = 438
    grayimgCrop = grayimg1[y-30:y+30,x-30:x+30]

    p00 = (x,y)
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    p0 = cv2.goodFeaturesToTrack(grayimg1, mask=None,
                                 **feature_params)

    lk_params = dict(winSize=(50, 50),
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))


    p00=p0.astype(int)
    pM  =   numpy.array([[452.0, 440.0]])
    pM = numpy.float32(pM)
    print((p0[0]).dtype)
    print(pM.dtype)

    #pM =cv2.KeyPoint(x=x,y=y, size=3, angle=0, response=0, octave=0, class_id=0)


    show1 =cv2.drawMarker(img1, pM[0].astype(int) , color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

    # calculate optical flow
    prev_pts = list()
    for r in range(grayimgCrop.shape[0]):
        for c in range(grayimgCrop.shape[1]):
            prev_pts.append((c, r))
    prev_pts = numpy.array(prev_pts, dtype=numpy.float32)

    p1, st, err = cv2.calcOpticalFlowPyrLK(grayimg1,
                                           grayimg2,
                                           pM, None,
                                           **lk_params)
    #flow = cv2.calcOpticalFlowFarneback(grayimgCrop,grayimg2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #p0 = p0[10].astype(int)



    #show1 =cv2.drawMarker(grayimgCrop, p0[0] , color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    #show2 =cv2.drawMarker(grayimg2, p0[0].astype(int), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

    # Create some random colors
    # Create a mask image for drawing purposes
    mask = numpy.zeros_like(img1)
    color = numpy.random.randint(0, 255, (100, 3))

    # Select good points
    """"
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(img1, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)"""
    # good_new = p1[st == (1)]
    # good_new = p1.astype(int)
    # for i in good_new:

    show2 =cv2.drawMarker(img2, p1[0].astype(int) , color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('imag1', show1)
    cv2.imshow('image2', show2)

    cv2.waitKey(0)  # wait for a keyboard input
    cv2.destroyAllWindows()



