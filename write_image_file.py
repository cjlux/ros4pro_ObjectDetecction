import cv2, rospy
import matplotlib.pyplot as plt
import matplotlib
from poppy_controllers.srv import GetImage
from cv_bridge import CvBridge
matplotlib.use('TkAgg')

i=1
while True:
    get_image = rospy.ServiceProxy("get_image", GetImage)
    response  = get_image()
    bridge    = CvBridge()
    image     = bridge.imgmsg_to_cv2(response.image)
    rep = input("press ENTER to write image in file [Q for quit] ")
    if rep.lower() == 'q': 
        break
    cv2.imwrite(f"image{i:03d}.png", image)
    i += 1

