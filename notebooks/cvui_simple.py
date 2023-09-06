import numpy as np
import cv2
import cvui
WINDOW_NAME = 'CVUI Test'
cvui.init(WINDOW_NAME)
frame = cv2.imread('./images/0_cam_image_array_.jpg')
while True:
    cvui.imshow(WINDOW_NAME, frame)
    cvui.update()
    if cv2.waitKey(1) == 97:
        break


