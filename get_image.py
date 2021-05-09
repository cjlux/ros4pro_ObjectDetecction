import cv2, rospy
import matplotlib.pyplot as plt
import matplotlib
from poppy_controllers.srv import GetImage
from cv_bridge import CvBridge
matplotlib.use('TkAgg')

get_image = rospy.ServiceProxy("get_image", GetImage)
response  = get_image()
bridge    = CvBridge()
image     = bridge.imgmsg_to_cv2(response.image)
plt.figure()
plt.imshow(image)
plt.axis('off')
plt.show()
cv2.imwrite("image.png", image)

