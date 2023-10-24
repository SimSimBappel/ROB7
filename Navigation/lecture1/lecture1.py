import matplotlib.pyplot as plt
import math
import numpy as np
from controller import Robot, Compass, Motor, Supervisor, Lidar#, LidarPoint


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
        
    return bearing
    

#get simulation heading
def simHeading():

    # translation = trans.getSFVec3f()
    # rotation  = rot.getSFRotation()
    
    orientation = tiago_node.getOrientation()
    
    # Extract yaw angle (rotation around z-axis)
    yaw = math.atan2(orientation[1], orientation[0])
    


    # Convert yaw angle to degrees
    yaw_degrees = math.degrees(yaw)
    
    yaw_degrees = (yaw_degrees + 360) % 360
  
    return yaw_degrees
    
    
    

    
    

timestep = int(robot.getBasicTimeStep())


# Compass
compass = Compass("compass")

if compass is None:

    print("Could not find a compass.")

    exit()

compass.enable(10)



# MOTORS

# Find motors and set them up to velocity control

left_motor = Motor("wheel_left_joint")

right_motor = Motor("wheel_right_joint")



if left_motor is None or right_motor is None:

    print("Could not find motors.")

    exit()

left_motor.setPosition(float("inf"))

right_motor.setPosition(float("inf"))


count = 0
errors = []

while robot.step(timestep) != -1:


    left_motor.setVelocity(1.0)

    right_motor.setVelocity(0.0)
    
    heading = getBearing()
    simheading = simHeading()
    error = heading - simheading
    errors.append(error)
    count += 1
    
    print(f"Compass: {heading}, simulation: {simheading}, error: {error}")
    
    if heading > 350 and heading < 355:
        plt.clf()
        plt.scatter(range(len(errors)), errors, s=1)
        plt.title("error = compass - simulation")
        errors = []
        count = 0
        plt.show()
        
    
    
    

