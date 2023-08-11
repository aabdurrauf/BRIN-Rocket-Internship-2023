from gym import spaces
import numpy as np
from time import sleep
import xpc

class XPlane11:
    def __init__(self, xpc_client):
        self.observation_space = spaces.Box(-np.inf, +np.inf, (6,))
        self.action_space = [0, 0, 0, 0]
        self.state = []
        self.untransformed_state = [0] * 6
        self.action_history = []
        self.prev_shaping = None
        self.is_on_ground = 0

        self.client = xpc_client

        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            self.client.getDREF("sim/test/test_float")
        except Exception as e:
            print("Error establishing connection to X-Plane.")
            self.client.close()
            self.client = None
            return

        print("Connection established")
        self._set_starship()

    def _set_starship(self):
        X = -998
        # set rocket altitude 1000 meters above sea level
        values = [X, X, 1000, X, X, X, X]
        self.client.sendPOSI(values, 0)
        # set roll rate to 50 degrees/second
        client.sendDREF('sim/flightmodel/position/P', 50)

    def _step(self, action):
        assert len(action) == 4
        self.CTRL = client.getCTRL(0)
        elevator = self.CTRL[0]
        aileron = self.CTRL[1]
        rudder = self.CTRL[2]
        throttle = self.CTRL[3]
        self.action_history.append([elevator, aileron, rudder, throttle])

        # State Vector
        self.previous_state = self.state  # Keep a record of the previous state
        state, self.untransformed_state = self.__generate_state()  # Generate state
        self.state = state  # Keep a record of the new state

        # set gear down if approaching ground
        if self.alti < 22.22 + 5:
            X = -998
            values = [X, X, X, X, X, X, 1]
            client.sendPOSI(values, 0)

        # check if crash
        crash = client.getDREF("sim/flightmodel2/misc/has_crashed")
        if crash[0]:
            # reward = −100000 in case the aircraft crashes
            reward = -100000
            done = True
        else:
            # Rewards for reinforcement learning
            reward = self.__compute_rewards(state)
            done = False
            if self.is_on_ground:  # self.alti <= 22.22
                done = True
        sleep(0.001)  # wait for 1 millisecond
        return np.array(state), reward, done, {}

    def __generate_state(self):
        self.alti = client.getDREF('sim/cockpit2/gauges/indicators/radio_altimeter_height_ft_pilot')[0]
        self.posx = client.getDREF('sim/flightmodel/position/local_x')[0]
        # pos y is the altitude of the rocket
        self.posz = client.getDREF('sim/flightmodel/position/local_z')[0]
        self.ver_vel = client.getDREF('sim/flightmodel/position/vh_ind')[0]
        self.pitch = client.getDREF('sim/flightmodel/position/theta')[0]
        self.roll = client.getDREF('sim/flightmodel/position/phi')[0]

        self.target = [-15531, 22.2156, -55350.6]
        # we have to normalize this state first using target
        state = [self.alti, self.posx, self.posz, self.ver_vel, self.pitch, self.roll]
        untransformed_state = [self.alti, self.posx, self.posz, self.ver_vel, self.pitch, self.roll]

        return state, untransformed_state

    def __compute_rewards(self, state):
        # state = [self.alti, self.posx, self.posz, self.ver_vel, self.pitch, self.roll]
        reward = 0
        # negative reward for every step and altitude
        shaping = -20
        # positive reward if land inside the landing area
        self.is_on_ground = client.getDREF('sim/flightmodel/failures/onground_any')[0]
        if self.is_on_ground:
            if (-2.75 * state[1] - 98154.25) <= state[2] <= (-2.75 * state[1] - 97968.25):
                if (0.35 * state[1] - 49946.15) <= state[2] <= (0.35 * state[1] - 49884.15):
                    shaping += 6000
                    print("rocket is inside the lander")
                else:
                    # penalize if landed outside the landing area
                    shaping -= 6000
                    print("rocket is outside the lander")
            else:
                shaping -= 6000
                print("rocket is outside the lander")
        else:
            shaping -= np.sqrt(state[0] - 22.22)
        # penalize for increasing vertical velocity
        if state[3] > 0:
            shaping -= state[3] * 100
        # keep track of previous reward shaping
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        return reward


client = xpc.XPlaneConnect()
xPlane11 = XPlane11(client)
client.close()
