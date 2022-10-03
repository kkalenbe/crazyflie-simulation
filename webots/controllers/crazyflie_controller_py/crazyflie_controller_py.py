#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 
# MIT License

# Copyright (c) 2022 Bitcraze

# @file crazyflie_controllers_py.py
# Controls the crazyflie motors in webots in Python

"""crazyflie_controller_py controller."""


from controller import Robot
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Gyro
from controller import Keyboard
from controller import Camera
from controller import RangeFinder

from synthetic_data_processing import process_camera_image, process_tof_image

import cv2
import numpy as np
import random
random.seed()

from math import cos, sin

import sys
sys.path.append('../../../controllers/')
from  pid_controller import init_pid_attitude_fixed_height_controller, pid_velocity_fixed_height_controller
from pid_controller import MotorPower_t, ActualState_t, GainsPID_t, DesiredState_t
robot = Robot()

timestep = int(robot.getBasicTimeStep())

## Initialize motors
m1_motor = robot.getDevice("m1_motor");
m1_motor.setPosition(float('inf'))
m1_motor.setVelocity(-1)
m2_motor = robot.getDevice("m2_motor");
m2_motor.setPosition(float('inf'))
m2_motor.setVelocity(1)
m3_motor = robot.getDevice("m3_motor");
m3_motor.setPosition(float('inf'))
m3_motor.setVelocity(-1)
m4_motor = robot.getDevice("m4_motor");
m4_motor.setPosition(float('inf'))
m4_motor.setVelocity(1)

## Initialize Sensors
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
Keyboard().enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)
tof = robot.getDevice("tof_matrix")
tof.enable(timestep)
    
## Initialize variables
actualState = ActualState_t()
desiredState = DesiredState_t()
pastXGlobal = 0
pastYGlobal = 0
past_time = robot.getTime()

## Initialize PID gains.
gainsPID = GainsPID_t()
gainsPID.kp_att_y = 1
gainsPID.kd_att_y = 0.5
gainsPID.kp_att_rp =0.5
gainsPID.kd_att_rp = 0.1
gainsPID.kp_vel_xy = 2
gainsPID.kd_vel_xy = 0.5
gainsPID.kp_z = 10
gainsPID.ki_z = 50
gainsPID.kd_z = 5
init_pid_attitude_fixed_height_controller()

## Speeds
forward_speed = 0.8
yaw_rate = 1.0

## Avoidance state
avoid_yawDesired = 0
avoid_yawTime = 0

## Initialize struct for motor power
motorPower = MotorPower_t()

print('Take off!')

heightDesired = 1.0

# Main loop:
while robot.step(timestep) != -1:

    dt = robot.getTime() - past_time;

    ## Get measurements
    actualState.roll = imu.getRollPitchYaw()[0]
    actualState.pitch = imu.getRollPitchYaw()[1]
    actualState.yaw_rate = gyro.getValues()[2];
    actualState.altitude = gps.getValues()[2];
    xGlobal = gps.getValues()[0]
    vxGlobal = (xGlobal - pastXGlobal)/dt
    yGlobal = gps.getValues()[1]
    vyGlobal = (yGlobal - pastYGlobal)/dt

    ## Get body fixed velocities
    actualYaw = imu.getRollPitchYaw()[2];
    cosyaw = cos(actualYaw)
    sinyaw = sin(actualYaw)
    actualState.vx = vxGlobal * cosyaw + vyGlobal * sinyaw
    actualState.vy = - vxGlobal * sinyaw + vyGlobal * cosyaw

    ## Initialize setpoints
    desiredState.roll = 0
    desiredState.pitch = 0
    desiredState.vx = 0
    desiredState.vy = 0
    desiredState.yaw_rate = 0
    desiredState.altitude = 2.0

    forwardDesired = 0
    sidewaysDesired = 0
    yawDesired = 0

    ## Get camera image
    w, h = camera.getWidth(), camera.getHeight()
    cameraData = camera.getImage()  # Note: uint8 string
    cameraImageRaw = np.copy(np.frombuffer(cameraData, np.uint8).reshape(h, w, 4))

    ## Get tof image
    w, h = tof.getWidth(), tof.getHeight()
    tofData = tof.getRangeImage(data_type="buffer")
    tofImageRaw = np.copy(np.frombuffer(tofData, np.float32).reshape(h, w, 1))

    # Process tof and camera images to simulate real sensors
    cameraImageProcessed = process_camera_image(cameraImageRaw)
    tofImageProcessed = process_tof_image(tofImageRaw)

    # Scale the tof images from 0.0-3.0 into 0-255
    tofImageRaw = (tofImageRaw * 255 / 3).astype(np.uint8)
    tofImageProcessed = (tofImageProcessed * 255 / 3).astype(np.uint8)

    # For visibility of output only
    scale = 50
    tofImageRaw = cv2.resize(tofImageRaw, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST);
    tofImageProcessed = cv2.resize(tofImageProcessed, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST);

    # Show raw images
    cv2.imshow('Drone camera raw', cameraImageRaw)
    cv2.waitKey(1)
    cv2.imshow('Drone tof raw', tofImageRaw)
    cv2.waitKey(1)

    # Show processed images
    cv2.imshow('Drone camera processed', cameraImageProcessed)
    cv2.waitKey(1)
    cv2.imshow('Drone tof processed', tofImageProcessed)
    cv2.waitKey(1)


    # Manual override
    key = Keyboard().getKey()
    while key>0:
        if key == Keyboard.UP:
            forwardDesired = forward_speed
        elif key == Keyboard.DOWN:
            forwardDesired = -forward_speed
        elif key == Keyboard.RIGHT:
            sidewaysDesired  = -forward_speed
        elif key == Keyboard.LEFT:
            sidewaysDesired = forward_speed
        elif key == ord('Q'):
            yawDesired = + yaw_rate
        elif key == ord('E'):
            yawDesired = - yaw_rate
        elif key == ord('W'):
            heightDesired += 0.01
        elif key == ord('S'):
            heightDesired -= 0.01

        key = Keyboard().getKey()

    desiredState.yaw_rate = yawDesired;

    ## PID velocity controller with fixed height
    desiredState.vy = sidewaysDesired;
    desiredState.vx = forwardDesired;
    desiredState.altitude = heightDesired;
    pid_velocity_fixed_height_controller(actualState, desiredState, gainsPID, dt, motorPower);

    m1_motor.setVelocity(-motorPower.m1)
    m2_motor.setVelocity(motorPower.m2)
    m3_motor.setVelocity(-motorPower.m3)
    m4_motor.setVelocity(motorPower.m4)
    
    past_time = robot.getTime()
    pastXGlobal = xGlobal
    pastYGlobal = yGlobal
