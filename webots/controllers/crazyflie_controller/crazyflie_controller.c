/* 
 *  ...........       ____  _ __
 *  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 *  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 *  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *  
 * MIT License
 * 
 * Copyright (c) 2022 Bitcraze
 * 
 * @file crazyflie_controller.c
 * Controls the crazyflie motors in webots
 */

#include <math.h>
#include <stdio.h>

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/gps.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>

#include "../../../controllers/pid_controller.h"

int main(int argc, char **argv) {
  wb_robot_init();

  int timestep = (int)wb_robot_get_basic_time_step();

  // Initialize motors
  WbDeviceTag m1_motor = wb_robot_get_device("m1_motor");
  wb_motor_set_position(m1_motor, INFINITY);
  wb_motor_set_velocity(m1_motor, -1.0);
  WbDeviceTag m2_motor = wb_robot_get_device("m2_motor");
  wb_motor_set_position(m2_motor, INFINITY);
  wb_motor_set_velocity(m2_motor, 1.0);
  WbDeviceTag m3_motor = wb_robot_get_device("m3_motor");
  wb_motor_set_position(m3_motor, INFINITY);
  wb_motor_set_velocity(m3_motor, -1.0);
  WbDeviceTag m4_motor = wb_robot_get_device("m4_motor");
  wb_motor_set_position(m4_motor, INFINITY);
  wb_motor_set_velocity(m4_motor, 1.0);

  // Initialize Sensors
  WbDeviceTag imu = wb_robot_get_device("inertial unit");
  wb_inertial_unit_enable(imu, timestep);
  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, timestep);
  wb_keyboard_enable(timestep);

  // Wait for 2 seconds
  while (wb_robot_step(timestep) != -1) {
    if (wb_robot_get_time() > 2.0)
      break;
  }

  // Initialize variables
  ActualState_t actualState = {0};
  DesiredState_t desiredState = {0};
  double pastXGlobal =0;
  double pastYGlobal=0;
  double past_time = wb_robot_get_time();
  double yawDesired = 0;

  // Initialize PID gains.
  GainsPID_t gainsPID;
  gainsPID.kp_att_y = 1;
  gainsPID.kd_att_y = 0.5;
  gainsPID.kp_att_rp =0.5;
  gainsPID.kd_att_rp = 0.1;
  gainsPID.kp_vel_xy = 2;
  gainsPID.kd_vel_xy = 0.5;
  gainsPID.kp_z = 10;
  gainsPID.ki_z = 50;
  gainsPID.kd_z = 5;
  init_pid_attitude_fixed_height_controller();

  // Initialize struct for motor power
  MotorPower_t motorPower;

  printf("Take off!\n");

  while (wb_robot_step(timestep) != -1) {

    const double dt = wb_robot_get_time() - past_time;

    // Get measurements
    actualState.roll = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
    actualState.pitch = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
    actualState.yaw = wb_inertial_unit_get_roll_pitch_yaw(imu)[2];
    actualState.altitude = wb_gps_get_values(gps)[2];
    double xGlobal= wb_gps_get_values(gps)[0];
    double vxGlobal = (xGlobal - pastXGlobal)/dt;
    double yGlobal = wb_gps_get_values(gps)[1];
    double vyGlobal = (yGlobal - pastYGlobal)/dt;

    // Get body fixed velocities
    double cosyaw = cos(actualState.yaw);
    double sinyaw = sin(actualState.yaw);
    actualState.vx = vxGlobal * cosyaw + vyGlobal * sinyaw;
    actualState.vy = - vxGlobal * sinyaw + vyGlobal * cosyaw;

    // Initialize values
    desiredState.roll = 0;
    desiredState.pitch = 0;
    desiredState.vx = 0;
    desiredState.vy = 0;
    desiredState.yaw = 0;
    desiredState.altitude = 2.0;

    double forwardDesired = 0;
    double sidewaysDesired = 0;

    // Control altitude
    int key = wb_keyboard_get_key();
    while (key > 0) {
      switch (key) {
        case WB_KEYBOARD_UP:
          forwardDesired = + 1.0;
          break;
        case WB_KEYBOARD_DOWN:
          forwardDesired = - 1.0;
          break;
        case WB_KEYBOARD_RIGHT:
          sidewaysDesired = - 1.0;
          break;
        case WB_KEYBOARD_LEFT:
          sidewaysDesired = + 1.0;
          break;
        case 'Q':
          yawDesired = actualState.yaw + 0.15;
          break;
        case 'E':
          yawDesired = actualState.yaw - 0.15;
          break;
        }
      key = wb_keyboard_get_key();
    }


    desiredState.yaw = yawDesired;

    // PID velocity controller with fixed height
    desiredState.vy = sidewaysDesired;
    desiredState.vx = forwardDesired;
    pid_velocity_fixed_height_controller(actualState, &desiredState,
    gainsPID, dt, &motorPower);

    // PID attitude controller with fixed height
    /*desiredState.roll = sidewaysDesired;
    desiredState.pitch = forwardDesired;
     pid_attitude_fixed_height_controller(actualState, &desiredState,
    gainsPID, dt, &motorPower);*/
    
    // Setting motorspeed
    wb_motor_set_velocity(m1_motor, - motorPower.m1);
    wb_motor_set_velocity(m2_motor, motorPower.m2);
    wb_motor_set_velocity(m3_motor, - motorPower.m3);
    wb_motor_set_velocity(m4_motor, motorPower.m4);
    
    // Save past time for next time step
    past_time = wb_robot_get_time();
    pastXGlobal = xGlobal;
    pastYGlobal = yGlobal;


  };

  wb_robot_cleanup();

  return 0;
}
