from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import xpc
import math


def launch():
    print("Setting up simulation")
    with xpc.XPlaneConnect() as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return

        # increasing the throttle
        print("increasing the throttle")
        throttle = 1
        ctrl = [0.0, 0.0, 0.0, throttle]
        client.sendCTRL(ctrl)

        # initialize arrays
        dt = 0.1
        end_t = 25
        throttle_graph = []
        altitude_graph = []
        ver_vel_graph = []
        pitch_graph = []
        roll_graph = []
        elevator_graph = []

        # desired values
        des_altitude = 10
        des_velocity = 5
        des_pitch = 0
        des_roll = 0

        count_reward = False
        is_falling = False
        max_throttle_height = 500
        pitch_integral = 0
        roll_integral = 0

        ################################################################
        ## Reinforcement Learning ##
        # calculate the reward
        # ∗ reward = −20 for each timestep
        # ∗ reward = +6000 for being inside the target zone
        # ∗ reward = −sqrt |current altitude − target altitude|
        # ∗ reward = −100000 in case the aircraft crashes
        # ∗ reward = −20000 if the episode and ends and the aircraft was not in the target zone
        # TODO
        # question: is this a summation and subtraction or
        #           just changing according to the corresponding state?
        reward = 0
        ################################################################

        while True:
            try:
                # veloDREF = 'sim/cockpit2/gauges/indicators/vvi_fpm_pilot'
                # vertical_velocity = client.getDREF(veloDREF)[0] * 0.00508
                veloDREF = 'sim/flightmodel/position/vh_ind'
                vertical_velocity = client.getDREF(veloDREF)[0]
                ver_vel_graph.append(vertical_velocity)

                altiDREF = 'sim/cockpit2/gauges/indicators/radio_altimeter_height_ft_pilot'
                altitude = client.getDREF(altiDREF)[0] * 0.3048
                altitude_graph.append(altitude)

                # pitchDREF = 'sim/cockpit2/gauges/indicators/pitch_AHARS_deg_pilot'
                pitchDREF = 'sim/flightmodel/position/theta'
                pitch = client.getDREF(pitchDREF)[0]
                pitch_graph.append(pitch)

                p_rateDREF = 'sim/flightmodel/position/Q'
                pitch_rate = client.getDREF(p_rateDREF)[0]

                # rollDREF = 'sim/cockpit2/gauges/indicators/roll_AHARS_deg_pilot'
                rollDREF = 'sim/flightmodel/position/phi'
                roll = client.getDREF(rollDREF)[0]
                roll_graph.append(roll)

                r_rateDREF = 'sim/flightmodel/position/P'
                roll_rate = client.getDREF(r_rateDREF)[0]

                print("Altitude:", altitude, "\nVer. Velocity:", vertical_velocity,
                      "\nPitch Angle:", pitch, "\nRoll Angle:", roll, "\n")

            except:
                ver_vel_graph.append(0)
                altitude_graph.append(0)
                pitch_graph.append(0)
                roll_graph.append(0)
                print("error occurred")

            if not is_falling and altitude > max_throttle_height:
                is_falling = True
                throttle = 0

            if not count_reward and is_falling and vertical_velocity <= 0:
                count_reward = True
                print("Activating RL")

            altitude_err = abs(des_altitude - altitude)
            velocity_err = vertical_velocity - des_velocity
            pitch_err = des_pitch - pitch
            roll_err = des_roll - roll

            Pp = 0.054
            Ip = 0.0012
            Dp = 0.001

            Pr = 0.02
            Ir = 0
            Dr = 0.002

            if is_falling and altitude < max_throttle_height:
                throttle = pow(2.71828, (34.2 - 0.084 * altitude))
                if abs(vertical_velocity) < 3:
                    throttle = 0
                    gearDREF = 'sim/cockpit/switches/gear_handle_status'
                    client.sendDREF(gearDREF, 1)
                    is_falling = False

                # change PD coefficient if falling down
                Pp = 0.04
                Dp = 0.0006
                Ip = 0
                Ir = 0

            pitch_integral += pitch_err
            roll_integral += roll_err

            elevator = pitch_err * Pp + pitch_rate * Dp + pitch_integral * Ip
            if elevator > 1:
                elevator = 1
            if elevator < -1:
                elevator = -1

            aileron = roll_err * Pr + roll_rate * Dr + roll_integral * Ir
            if aileron > 1:
                aileron = 1
            if aileron < -1:
                aileron = -1

            if throttle > 1:
                throttle = 1

            elevator_graph.append(elevator)
            control_values = [elevator, aileron, 0, throttle]
            client.sendCTRL(control_values)
            throttle_graph.append(throttle)

            sleep(dt)
            # calculate the reward
            # ∗ reward = −20 for each timestep
            # ∗ reward = +6000 for being inside the target zone
            # ∗ reward = −sqrt |current altitude − target altitude|
            # ∗ reward = −100000 in case the aircraft crashes
            # ∗ reward = −20000 if the episode and ends and the aircraft was not in the target zone
            #   target altitude = des_altitude
            if count_reward:
                reward = reward - 20 - math.sqrt(abs(altitude - des_altitude))
                if altitude < 1:  # if the rocket hit the target altitude and position
                    reward += 6000
                print("reward:", reward)

            crash = client.getDREF("sim/flightmodel2/misc/has_crashed")
            # print(crash)
            if crash[0] == 1:
                # ∗ reward = −100000 in case the aircraft crashes
                reward -= 100000
                print("crash")
                break

        # setting the time
        t = np.linspace(0, len(altitude_graph) / 10 - 1, len(altitude_graph))
        # plotting the data
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(t, altitude_graph)
        axs[0, 0].set_title("Altitude (m) vs Time")
        axs[0, 0].grid()
        axs[0, 1].plot(t, ver_vel_graph)
        axs[0, 1].set_title("Vertical Velocity (m/s) vs Time")
        axs[0, 1].grid()
        axs[1, 0].plot(t, pitch_graph)
        axs[1, 0].set_title("Pitch Angle (degree) vs Time")
        axs[1, 0].grid()
        axs[1, 1].plot(t, roll_graph)
        axs[1, 1].set_title("Roll Angle (degree) vs Time")
        axs[1, 1].grid()
        axs[2, 0].plot(t, throttle_graph)
        axs[2, 0].set_title("Throttle vs Time")
        axs[2, 0].grid()
        axs[2, 1].plot(t, elevator_graph)
        axs[2, 1].set_title("Elevator vs Time")
        axs[2, 1].grid()

        fig.tight_layout()  # adjust subplot parameters to give specified padding
        plt.show()

    return reward


if __name__ == "__main__":
    reward = launch()
    print(reward)
