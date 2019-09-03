import numpy as np
import cv2
boundaries = [
   ([160, 100, 49], [177, 255, 255]),
    ([0, 0, 0], [0, 0, 0])
]
#([160, 83, 80], [180, 255, 255]),

def handsegment(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask1 = cv2.inRange(frame, lower, upper)

    lower, upper = boundaries[1]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask2 = cv2.inRange(frame, lower, upper)

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
    mask = cv2.bitwise_or(mask1, mask2)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    output = cv2.bitwise_and(frame, frame, mask=mask)

    # show the images
    #cv2.imshow("images", output)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return output

if __name__ == '__main__':
    frame = cv2.imread("test.jpeg")
    handsegment(frame)
