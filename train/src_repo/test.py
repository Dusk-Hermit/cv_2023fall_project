# Import necessary libraries
import cv2
import numpy as np

# Read an image
image_width = 640
image_height = 480

# 创建一个包含零值的数组
img = np.zeros((image_height, image_width, 3), dtype=np.uint8)


# Define an array of endpoints of Hexagon
points = np.array([[220, 120], [130, 200], [130, 300],
				[220, 380], [310, 300], [310, 200]])

# Use fillPoly() function and give input as image,
# end points,color of polygon
# Here color of polygon will be green
cv2.fillPoly(img, pts=[points], color=(0, 255, 0))

# Displaying the image
cv2.imshow("Hexagon", img)

# wait for the user to press any key to 
# exit window
cv2.waitKey(0)

# Closing all open windows
cv2.destroyAllWindows()
