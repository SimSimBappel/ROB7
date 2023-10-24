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
    rad = math.atan2(north[1], north[0])
    bearing = (rad - 1.5708) / math.pi * 180.0
    
    if bearing < 0.0:
        bearing = bearing + 360.0
        
    return bearing, rad - np.pi/2
    

#get simulation heading
def ass():
    translation = trans.getSFVec3f()
    rotation  = rot.getSFRotation()
    
    #r,p,y = euler_from_quaternion(rotation)
    
    # yaw = math.degrees(y) + 180
    
    # yaw = math.degrees(rotation[3])
    
    yaw = rotation[3] * (180 / math.pi)

    
    return yaw % 360
    
    

    
    

timestep = int(robot.getBasicTimeStep())


# Compass
compass = Compass("compass")

if compass is None:

    print("Could not find a compass.")

    exit()

compass.enable(10)


# lidar
# lidar = Lidar("lidar")

# if lidar is None:
    # print("could not find lidar")
    # exit(3)

lidar = robot.getDevice("lidar")
lidar.enable(100) 
lidar.enablePointCloud()

print(f"LIDAR: min range: {lidar.getMinRange()}, max range: {lidar.getMaxRange()}, pointcloud enabled: {lidar.isPointCloudEnabled()}, Hresolution: {lidar.getHorizontalResolution()}")





# MOTORS

# Find motors and set them up to velocity control

left_motor = Motor("wheel_left_joint")

right_motor = Motor("wheel_right_joint")



if left_motor is None or right_motor is None:

    print("Could not find motors.")

    exit()

left_motor.setPosition(float("inf"))

right_motor.setPosition(float("inf"))




while robot.step(timestep) != -1:


    left_motor.setVelocity(1.0)

    right_motor.setVelocity(0.0)
    
    
    
    # data = lidar.getPointCloud()
    # data = lidar.getLayerPointCloud(1) #datatype list is slow
    # data = lidar.getLayerRangeImage(1)
    
    # print(data)


    # print(data)
    
    range_image = lidar.getLayerRangeImage(1)
    #print(range_image)
    range_image.reverse()
    x = [0]
    y = [0]
    #print(lidar.getFov())
    # Print out the LiDAR points
    heading, rad = getBearing()
    # print(np.degrees(rad))
    for index, distance in enumerate(range_image):
        angle = rad + lidar.getFov() / 2 - index * lidar.getFov() / (len(range_image) - 1)
        x.append(distance * np.sin(angle))
        y.append(distance * np.cos(angle))
            
        # print(f"pointHead: {np.degrees(pointHeading)}, rad: {np.degrees(rad)}, compass: {heading},index: {index}, distance: {distance}")
        # print(f"x: {distance * np.sin(pointHeading)}, y: {distance * np.cos(pointHeading)}")
            
  
    plt.clf()
    plt.scatter(x, y, s=1)
    plt.xlim(-5, 5)  # Adjust limits as needed
    plt.ylim(-5, 5)  # Adjust limits as needed
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pause(0.01)

    
    
plt.show()
   

