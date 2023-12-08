import rospy
from std_msgs.msg import Float64MultiArray
import time

if __name__ == '__main__':
    rospy.init_node('FakePublisher')

    freq = 20
    rate = rospy.Rate(freq)

    fake_pub = rospy.Publisher('fakeParameters',Float64MultiArray,queue_size=1)
    params = Float64MultiArray()

    params.data = [100,2,0,0,0,0,11.25,20.25,100,2,0,0,0,0,11.25,10.25]

    while not rospy.is_shutdown():
        fake_pub.publish(params)
        rate.sleep()
