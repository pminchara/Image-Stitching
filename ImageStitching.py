import cv2
import numpy as np
print ("OpenCV Version: ",cv2.__version__)

# The picture used in this project was captured by Inchara Polepalli Muneshkumar in Dublin.

#Left part of image
imLeft = cv2.imread('../parts/left_image.jpg')
imLeft = cv2.resize(imLeft,(550,800),None,0.1,0.1)
imLeftGray = cv2.cvtColor(imLeft,cv2.COLOR_RGB2GRAY)

#Right part of image
imRight = cv2.imread('../parts/right_image.jpg')
imRight = cv2.resize(imRight,(550,800),None,0.1,0.1)
imRightGray = cv2.cvtColor(imRight,cv2.COLOR_RGB2GRAY)

#SIFT -> Scale-Invariant Feature Transform
#Using SIFT from OpenCV
sift = cv2.xfeatures2d.SIFT_create()

# Keypoint detection
key_pt_left, des_left = sift.detectAndCompute(imLeft,None)
key_pt_right, des_right = sift.detectAndCompute(imRight,None)

cv2.imshow('Key points in Left Image', cv2.drawKeypoints(imLeft,key_pt_left,None))
cv2.imshow('Key points in Right Image', cv2.drawKeypoints(imRight,key_pt_right,None))

# Using Matcher method

match = cv2.BFMatcher()
matches = match.knnMatch(des_right, des_left, k=2)

fit=[]
for m,n in matches:
    if(m.distance < 0.75 * n.distance):
        fit.append([m])


draw_line = dict(matchColor = (0,255,255),flags=2)
imMatch =cv2.drawMatchesKnn(imRight, key_pt_right, imLeft, key_pt_left, fit, None, **draw_line)
cv2.imshow("Matching depiction", imMatch)
cv2.imwrite("../match/Matches.jpg", imMatch)


def warpTwoImages(im1, im2, H):
    
    height_1, width_1 = im1.shape[:2]
    height_2, width_2 = im2.shape[:2]
    
    points1 = np.float32([[0,0],[0,height_1],[width_1,height_1],[width_1,0]]).reshape(-1,1,2)
    points2 = np.float32([[0,0],[0,height_2],[width_2,height_2],[width_2,0]]).reshape(-1,1,2)

    points2_ = cv2.perspectiveTransform(points2, H)
    points = np.concatenate((points1, points2_), axis=0)
    
    [xmin, ymin] = np.int32(points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(points.max(axis=0).ravel() + 0.5)
    
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    stitch = cv2.warpPerspective(im2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    stitch[t[1]:height_1+t[1],t[0]:width_1+t[0]] = im1
    return stitch


# Homography graph Best matching points
MIN_MATCH_COUNT = 10

if len(fit) > MIN_MATCH_COUNT:
    
    src_points = np.float32([key_pt_right[m[-1].queryIdx].pt for m in fit]).reshape(-1,1,2)
    des_points = np.float32([key_pt_left[m[-1].trainIdx].pt for m in fit]).reshape(-1,1,2)
    
    M, mask  = cv2.findHomography(src_points, des_points, cv2.RANSAC,5.0)
    pan_Image = warpTwoImages(imLeft, imRight, M)
    
    cv2.imshow("Panaromic view", pan_Image)
    cv2.imwrite("../panoramic_view/final_image.jpg", pan_Image)

else:
    print("Not Enough Matches")

cv2.waitKey(0)
cv2.destroyAllWindows()