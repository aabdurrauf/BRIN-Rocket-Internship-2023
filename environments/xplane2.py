import gym
from gym import spaces
import numpy as np
from time import sleep
import xpc


class XPlane11(gym.Env):
    def __init__(self, xpc_client):
        self.observation_space = spaces.Box(-np.inf, +np.inf, (4,))
        self.action_space = [0]  # only throttle as action space
        self.state = []
        self.untransformed_state = [0] * 4
        self.prev_shaping = None
        self.is_on_ground = 0
        self.CTRL = None
        self.previous_state = None
        self.action_history = []
        self.black_box = []
        self.crash = 0
        self.pitch_integral = 0
        self.roll_integral = 0

        self.client = xpc_client

        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            xpc_client.getDREF("sim/test/test_float")
        except Exception as e:
            print("Error establishing connection to X-Plane.")
            self.client.close()
            self.client = None
            return

        print("Connection established-xplane")
        # self._set_starship()

    def _reset(self, set_starship=True):
        self.action_history = []
        self.black_box = []
        self.pitch_integral = 0
        self.roll_integral = 0
        if set_starship:
            self._set_starship()

        return self._step(np.array([0]))[0]

    def _set_starship(self):
        X = -998
        # set starship altitude 1000 meters above sea level
        values = [X, X, 1000, X, X, X, 0]
        try:
            self.client.sendPOSI(values, 0)
        except Exception as e:
            pass

    def _step(self, action):
        assert len(action) == 1

        # check if crash
        try:
            value = self.client.getDREF("sim/flightmodel2/misc/has_crashed")
            self.crash = value[0]
        except Exception as e:
            self.crash = 0
            self.client.clearBuffer()

        if self.crash != 1.0 and self.crash != 0.0:
            print("Error in has_crash dataref. self.crash:", self.crash)
            self.client.clearBuffer()
            return np.array(self.previous_state), -10000, True, {}

        if self.crash == 1:
            print("starship crashed, detected inside step function")
            done = True
            reward = -100000
            self.crash = 0  # ?
            return np.array(self.state), reward, done, {}

        # Check for contact with the ground
        try:
            self.is_on_ground = self.client.getDREF('sim/flightmodel/failures/onground_any')[0]
        except Exception as e:
            self.is_on_ground = False
            self.client.clearBuffer()

        if self.is_on_ground != 1.0 and self.is_on_ground != 0.0:
            print("Error in is_on_ground dataref. self.is_on_ground:", self.is_on_ground)
            self.client.clearBuffer()
            return np.array(self.previous_state), -10000, True, {}

        try:
            get_ctrl_values = self.client.getCTRL()
        except Exception as e:
            get_ctrl_values = [0, 0, 0, 0, 0, 0, 0]
            self.client.clearBuffer()

        # PID controller for pitch and roll
        try:
            pitch = self.client.getDREF('sim/flightmodel/position/theta')[0]
            pitch_rate = self.client.getDREF('sim/flightmodel/position/Q')[0]
            roll = self.client.getDREF('sim/flightmodel/position/phi')[0]
            roll_rate = self.client.getDREF('sim/flightmodel/position/P')[0]
        except Exception as e:
            pitch = 0
            pitch_rate = 0
            roll = 0
            roll_rate = 0
            self.client.clearBuffer()

        # pitch
        Pp = 0.04
        Ip = 0
        Dp = 0.006
        # roll
        Pr = 0.02
        Ir = 0
        Dr = 0.02
        self.pitch_integral += pitch
        self.roll_integral += roll

        elevator = -pitch * Pp + pitch_rate * Dp + self.pitch_integral * Ip
        if elevator > 1:
            elevator = 1
        if elevator < -1:
            elevator = -1

        aileron = -roll * Pr + roll_rate * Dr + self.roll_integral * Ir
        if aileron > 1:
            aileron = 1
        if aileron < -1:
            aileron = -1

        X = -998
        # Shutdown all Engines upon contact with the ground
        if self.is_on_ground:
            action = [0]

        throttle = action[0]
        # Execute the action

        control_values = [elevator, aileron, X, throttle]
        try:
            self.client.sendCTRL(control_values)
        except Exception as e:
            pass

        self.action_history.append(get_ctrl_values)

        # State Vector
        self.previous_state = self.state  # Keep a record of the previous state
        state = self.__generate_state()  # Generate state
        self.state = state  # Keep a record of the new state

        # set gear down if approaching ground
        if not self.is_on_ground and self.alti < 22.22 + 5:
            X = -998
            values = [X, X, X, X, X, X, 1]
            try:
                self.client.sendPOSI(values, 0)
            except Exception as e:
                pass

        # Rewards for reinforcement learning
        reward = self.__compute_rewards(state)
        done = False

        if self.is_on_ground:  # self.alti <= 22.22
            done = True
        return np.array(state), reward, done, {}

    def __generate_state(self):
        try:
            self.alti = self.client.getDREF('sim/cockpit2/gauges/indicators/radio_altimeter_height_ft_pilot')[0]
            self.posx = self.client.getDREF('sim/flightmodel/position/local_x')[0]
            # pos y is the altitude of the starship
            self.posz = self.client.getDREF('sim/flightmodel/position/local_z')[0]
            self.ver_vel = self.client.getDREF('sim/flightmodel/position/vh_ind')[0]
            pitch = self.client.getDREF('sim/flightmodel/position/theta')[0]
            roll = self.client.getDREF('sim/flightmodel/position/phi')[0]
            # for black box
            vx = self.client.getDREF('sim/flightmodel/forces/vx_acf_axis')[0]
            vz = self.client.getDREF('sim/flightmodel/forces/vz_acf_axis')[0]
            yaw = self.client.getDREF('sim/flightmodel/position/psi')[0]
            # pitch_rate = self.client.getDREF('sim/flightmodel/position/Q')[0]
            # roll_rate = self.client.getDREF('sim/flightmodel/position/P')[0]
            # yaw_rate = self.client.getDREF('sim/flightmodel/position/R')[0]

        except Exception as e:
            # set values to 0 if error occurred in communication with x-plane
            self.alti = 0
            self.posx = 0
            self.posz = 0
            self.ver_vel = 0
            self.client.clearBuffer()

            pitch = 0
            roll = 0
            vx = 0
            vz = 0
            yaw = 0
            # pitch_rate = 0
            # roll_rate = 0
            # yaw_rate = 0
        print("altitude:", self.alti, " ver. velocity:", self.ver_vel)
        self.black_box.append([self.alti, self.posx, self.posz, self.ver_vel, vx, vz, pitch, roll, yaw])

        # self.target = [-15531, 22.2156, -55350.6]
        # we have to normalize this state first using target
        state = [self.alti, self.posx, self.posz, self.ver_vel]
        # untransformed_state = [self.alti, self.posx, self.posz, self.ver_vel, self.pitch, self.roll]

        return state  # , untransformed_state

    def __compute_rewards(self, state):
        # state = [self.alti, self.posx, self.posz, self.ver_vel]
        reward = 0

        # negative reward for every step and altitude
        shaping = -20

        # positive reward if land inside the landing area
        if self.is_on_ground and not self.crash:
            if abs(self.ver_vel) < 1:
                if (-2.75 * state[1] - 98154.25) <= state[2] <= (-2.75 * state[1] - 97968.25):
                    if (0.35 * state[1] - 49946.15) <= state[2] <= (0.35 * state[1] - 49884.15):
                        shaping += 6000
                        print("starship landed inside the lander")
                    else:
                        # penalize if landed outside the landing area
                        shaping -= 6000
                        print("starship landed/crashed outside the lander")
                else:
                    shaping -= 6000
                    print("starship landed/crashed outside the lander")
        else:
            shaping -= np.sqrt(abs(state[0] - 22.22))
            # pass

        # give reward if approaching the ground and if the ver vel is low
        # if state[3] < 0:
        #     shaping = shaping + 3000 - state[0] + state[3] * 10

        # penalty for increasing vertical velocity
        ## if state[3] > 0:
        ##     shaping -= state[3] * 10

        # reward for decreasing velocity under 3190 ft meters
        # if self.alti <= 3190:
        #     shaping = shaping + (100 + self.ver_vel * 2)

        # keep track of previous reward shaping
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        return reward

    def apply_wind_force(self, wind_force, t):
        for n in range(t):
            try:
                self.client.sendDREF('sim/operation/override/override_forces', 1)
                self.client.sendDREF('sim/flightmodel/forces/fside_total', wind_force)
                sleep(1)
                self.client.sendDREF('sim/operation/override/override_forces', 0)
            except Exception as e:
                pass

    def get_action_history(self):
        return self.action_history

    def get_black_box(self):
        return self.black_box


def get_state_sample(env, xpc_client, samples=100, normal_state=True, untransformed_state=True):
    X = -998  # set value to -998 to keep unchanged
    # set starship altitude 1000 meters above sea level
    values = [X, X, 1000, X, X, X, X]
    try:
        xpc_client.sendPOSI(values, 0)
        # set roll rate to 50 degrees/second
        xpc_client.sendDREF('sim/flightmodel/position/P', 50)
    except Exception as e:
        pass
    # sleep(2)
    # env = XPlane11(xpc_client)
    # env.reset()
    # env._set_starship()

    # crash = False
    state_samples = []
    # while not crash:
    while len(state_samples) < samples:
        throttle = np.random.uniform(0, 1)
        action = [throttle]
        return_value = env.step(action)
        s = return_value[0]
        r = return_value[1]
        done = return_value[2]
        info = return_value[3]

        if done:
            crash = 1
            while crash:
                try:
                    crash = env.client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
                except Exception as e:
                    crash = 1
                    env.client.clearBuffer()
            env._reset()
            # crash = True
        else:
            if normal_state:
                state_samples.append(s)
            else:
                print("state samples not appended")
            # else:
            #     state_samples.append(
            #         env.get_state_with_barge_and_landing_coordinates(untransformed_state=untransformed_state))
    # env.close()
    return state_samples

# client = xpc.XPlaneConnect()
# xPlane11 = XPlane11(client)
# xPlane11.apply_wind_force(2000000, 5)
#
# client.close()
