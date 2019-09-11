import numpy as np
import cv2
import os
boundaries = [
    ([160, 100, 80], [180, 255, 255]),
]

def handsegment(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.waitKey(0)
    # for i,(lower, upper) in enumerate(boundaries):
    # 	# create NumPy arrays from the boundaries
    # 	lower = np.array(lower, dtype = "uint8")
    # 	upper = np.array(upper, dtype = "uint8")

    # 	# find the colors within the specified boundaries and apply
    # 	# the mask
    # 	if(i==0):
    # 		print "Harish"
    # 		mask1 = cv2.inRange(frame, lower, upper)
    # 	else:
    # 		print "Aadi"
    # 		mask2 = cv2.inRange(frame, lower, upper)
    #output = cv2.bitwise_and(frame, frame, mask=mask)
    # show the images
    #cv2.imshow("images", output)
    return output

if __name__ == '__main__':
    filenames = os.listdir('photos/finish/new')
    for name in filenames:
        filename = "photos/finish/new/" + name
        frame = cv2.imread(filename)
        cv2.waitKey(0)
        mask = handsegment(frame)
        cv2.imwrite('photos/finish/' + name , mask)
