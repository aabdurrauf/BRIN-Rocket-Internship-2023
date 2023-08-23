import numpy as np



################################################################

def _step(self, action):
    assert len(action) == 4

    # Check for contact with the ground
    try:
        self.is_on_ground = self.client.getDREF('sim/flightmodel/failures/onground_any')[0]
    except Exception as e:
        self.is_on_ground = False
    print("is on ground:", self.is_on_ground)
    # Shutdown all Engines upon contact with the ground
    if self.is_on_ground:
        action = [0, 0, 0, 0]

    # Execute the action
    control_values = [action[0], action[1], action[2], action[3]]
    # print("Elevator:", action[0], "Aileron:", action[1], "Rudder:", action[2], "Throttle:", action[3])
    try:
        self.client.sendCTRL(control_values)

        # Gather Stats
        self.CTRL = self.client.getCTRL(0)
        elevator = self.CTRL[0]
        aileron = self.CTRL[1]
        rudder = self.CTRL[2]
        throttle = self.CTRL[3]
    except Exception as e:
        elevator = 0
        aileron = 0
        rudder = 0
        throttle = 0

    self.action_history.append([elevator, aileron, rudder, throttle])

    # State Vector
    self.previous_state = self.state  # Keep a record of the previous state
    state, self.untransformed_state = self.__generate_state()  # Generate state
    self.state = state  # Keep a record of the new state

    # set gear down if approaching ground
    if not self.is_on_ground and self.alti < 22.22 + 5:
        X = -998
        values = [X, X, X, X, X, X, 1]
        try:
            self.client.sendPOSI(values, 0)
        except Exception as e:
            pass
    # check if crash
    # self.client.sendDREF("sim/flightmodel2/misc/has_crashed", 0)
    try:
        self.crash = self.client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
        print("crash:", self.crash)
    except Exception as e:
        self.crash = 0

    if self.crash:
        # reward = −100000 in case the aircraft crashes
        reward = -100000
        done = True
        print("starship crashed")
    else:
        # Rewards for reinforcement learning
        reward = self.__compute_rewards(state)
        done = False
        if self.is_on_ground:  # self.alti <= 22.22
            done = True
    # sleep(0.001)  # wait for 1 millisecond ???
    return np.array(state), reward, done, {}