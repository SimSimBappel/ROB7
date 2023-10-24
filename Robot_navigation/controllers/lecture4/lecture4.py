import matplotlib.pyplot as plt
import math
import numpy as np
from controller import Robot, Compass, Motor, Supervisor, Lidar#, LidarPoint

import time

robot = Supervisor()

tiago_node = robot.getFromDef('TiagoBase')

if tiago_node is None:
    sys.exit(1)
    

trans = tiago_node.getField('translation')
rot = tiago_node.getField('rotation')


def getBearing():
    north = compass.getValues()
    rad = math.atan2(north[0], north[1])
    rad2 = math.atan2(north[1], north[0])
    bearing = (rad - 1.5708) / math.pi * 180.0
    
    if bearing < 0.0:
        bearing = bearing + 360.0
        
    if rad < 0:
        rad += np.pi * 2     

        
    return rad, rad2 - np.pi/2
    


def getSimHeading():
    translation = trans.getSFVec3f()
    rotation  = rot.getSFRotation()
    
    if rotation[3] < 0:
        rotation[3] += np.pi * 2

    #goes to 2pi and then counts down for some reason
    return rotation[3]
    
    

timestep = int(robot.getBasicTimeStep())


# Compass
compass = Compass("compass")

if compass is None:

    print("Could not find a compass.")

    exit()

compass.enable(10)


# Lidar
lidar = robot.getDevice("lidar")
lidar.enable(100) 
lidar.enablePointCloud()

print(f"LIDAR: min range: {lidar.getMinRange()}, max range: {lidar.getMaxRange()}, pointcloud enabled: {lidar.isPointCloudEnabled()}, Hresolution: {lidar.getHorizontalResolution()}")





# MOTORS
left_motor = Motor("wheel_left_joint")
right_motor = Motor("wheel_right_joint")


if left_motor is None or right_motor is None:

    print("Could not find motors.")

    exit()

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# Motor position sensors
left_position_sensor = left_motor.getPositionSensor()
left_position_sensor.enable(timestep)
right_position_sensor = right_motor.getPositionSensor()
right_position_sensor.enable(timestep)

left_prev_position = left_position_sensor.getValue()
right_prev_position = right_position_sensor.getValue()



# Init variables
wheelBase = 0.4244 #from URDF 0.2022 center to wheel, 0.04 depth of wheel
wheel_radius = 0.0985 #from URDF

x = 0.0
y = 0.0
theta = 0.0



while robot.step(timestep) != -1:    
    
    rad, rad2 = getBearing()
    
    left_position = left_position_sensor.getValue()
    right_position = right_position_sensor.getValue()

    # Calculate the distance traveled by each wheel
    left_distance = (left_position - left_prev_position) * wheel_radius
    right_distance = (right_position - right_prev_position) * wheel_radius

    # Calculate the change in orientation (theta)
    delta_theta = (right_distance - left_distance) / wheelBase

    # Calculate the average distance traveled
    delta_s = (left_distance + right_distance) / 2.0

    # Update position and orientation
    x += delta_s * np.cos(theta + delta_theta/2.0)
    y += delta_s * np.sin(theta + delta_theta/2.0)
    theta += delta_theta
    
    if theta > np.pi * 2:
        theta -= np.pi * 2
    
    # Store current wheel positions for next iteration
    left_prev_position = left_position
    right_prev_position = right_position
    
    
    # print(f'x: {"{:,.2f}".format(x)}, y: {"{:,.2f}".format(y)}, theta: {"{:,.2f}".format(theta)},', end="")
    # print(f' heading error: {"{:,.2f}".format(abs(theta)-abs(rad))}')
    # print(f'rad: {"{:,.2f}".format(rad)}, yaw: {"{:,.2f}".format(yaw)}')
    
    if robot.getTime() > 2.0:
        right_motor.setVelocity(1.0)
        left_motor.setVelocity(0.0)
        
        
        
        
    #lidar scan   
    range_image = lidar.getLayerRangeImage(1)
    range_image.reverse()
    lidarx = [0]
    lidary = [0]
    
    
    # Print out the LiDAR points
    for index, distance in enumerate(range_image):
        angle = rad2 + lidar.getFov() / 2 - index * lidar.getFov() / (len(range_image) - 1)
        lidarx.append(distance * np.sin(angle))
        lidary.append(distance * np.cos(angle))
 
    plt.clf()
    plt.scatter(lidarx, lidary, s=1)
    plt.xlim(-5, 5)  # Adjust limits as needed
    plt.ylim(-5, 5)  # Adjust limits as needed
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pause(0.01)

    
    
plt.show()
   

        