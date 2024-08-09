import numpy as np
import math

class Controller(object):
    def __init__(self, waypoints):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_frame = 0
        self._current_timestamp = 0
        self._start_control_loop = False
        self._set_throttle = 0
        self._set_brake = 0
        self._set_steer = 0
        self._waypoints = waypoints
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi
        self.v_prev = 0
        self.t_prev = 0
        self.e_prev = 0
        self.e_iprev = 0
        self.o_tprev = 0
        self.o_sprev = 0
        self.K_p = 1.0
        self.K_i = 0.1
        self.K_d = 0.01

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed
        self._current_timestamp = timestamp
        self._current_frame = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_parameters(self, K_p, K_i, K_d, desired_speed):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self._desired_speed = desired_speed

    def update_desired_speed(self):
        min_idx = 0
        min_dist = float("inf")
        desired_speed = 0

        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(
                np.array([self._waypoints[i][0] - self._current_x, self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints) - 1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Output values between 0.0 and 1.0
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Convert input_steer from -1~1 radian
        steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Output values between 0.0 and 1.0
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        x = self._current_x  # meter
        y = self._current_y  # meter
        yaw = self._current_yaw  # radian
        v = self._current_speed  # m/s
       
        v_desired = self._desired_speed  # m/s
        t = self._current_timestamp
        waypoints = self._waypoints  # x, y, v
        throttle_output = 0  # 0~1
        steer_output = 0  # -1.22~1.22(rad)
        brake_output = 0  # 0~1
        i_v = 0

        if self._start_control_loop:

            # Longitudinal controller (PID Controller)
            
            e = v_desired - v
            dt = t - self.t_prev

            throttle_output = 0
            brake_output = 0

            e_v = v_desired - v
            i_v = self.e_iprev + e_v * dt
            d_v = (e_v - self.e_prev) / dt

            accel = self.K_p * e_v + self.K_i * i_v + self.K_d * d_v

            if (accel > 0):
                # Restrict throttle_output to the range 0 to 1 (Use the tanh function to scale values from -1 to 1, add 1 to shift to 0 to 2, then divide by 2)
                throttle_output = (np.tanh(accel) + 1) / 2

                 # Limit output changes to prevent sudden variations, ensuring the change does not exceed 0.1
                if (throttle_output - self.o_tprev > 0.1):
                    throttle_output = self.o_tprev + 0.1

            else:
                throttle_output = 0

            # Lateral controller (Pure Pursuit)
            steer_output = 0
            L = 2.7  # meter - wheel base length

            x_c = x - L * np.cos(np.pi * yaw / 180) / 2
            y_c = y - L * np.sin(np.pi * yaw / 180) / 2

            for i in waypoints:
                dist = np.sqrt((i[0] - x_c) ** 2 + (i[1] - y_c) ** 2)

                if dist > 4:
                    target = i
                    break
                else:
                    target = waypoints[0]

            x_desired = target[0]
            y_desired = target[1]

            alpha = math.atan2(y_desired - y_c, x_desired - x_c) - np.pi * yaw / 180
            delta = math.atan2(2 * L * np.sin(alpha), dist)
            steer_output = delta

            # Setting output values
            self.set_throttle(throttle_output)
            self.set_steer(steer_output)
            self.set_brake(brake_output)

        self.v_prev = v
        self.t_prev = t

        self.e_prev = v_desired - v
        self.e_iprev = i_v
        self.o_tprev = throttle_output
        self.o_sprev = steer_output