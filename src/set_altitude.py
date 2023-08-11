from time import sleep
import xpc
import time

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

        is_on_ground = client.getDREF('sim/flightmodel/failures/onground_any')[0]
        print("is_on_ground", is_on_ground)
        altitude = client.getPOSI()[2]
        print("initial altitude:", altitude)
        X = -998
        values = [X, X, altitude + 15, X, X, X, 0]
        client.sendPOSI(values, 0)
        time.sleep(0.1)
        is_on_ground = client.getDREF('sim/flightmodel/failures/onground_any')[0]
        print("is_on_ground", is_on_ground)
        values = [X, X, X, X, X, X, 1]  # opening the gear down
        client.sendPOSI(values, 0)
        # time.sleep(1)
        is_on_ground = client.getDREF('sim/flightmodel/failures/onground_any')[0]
        if is_on_ground:
            print("yeaah is_on_ground", is_on_ground)

        fuelDREF = 'sim/flightmodel/weight/m_fuel1'
        fuel = client.getDREF(fuelDREF)[0] * 2.20462
        print("initial fuel:", fuel, "lbs")
        new_fuel = 650000/2.2046244
        # client.sendDREF(fuelDREF, new_fuel)
        fuel = client.getDREF(fuelDREF)[0] * 2.20462
        print("after changing fuel:", fuel, "lbs")

        posxDREF = 'sim/flightmodel/position/local_x'
        posyDREF = 'sim/flightmodel/position/local_y'
        poszDREF = 'sim/flightmodel/position/local_z'
        posx = client.getDREF(posxDREF)[0]
        posy = client.getDREF(posyDREF)[0]
        posz = client.getDREF(poszDREF)[0]
        print("pos x:", posx, "pos y:", posy, "pos z:", posz)
        # client.sendDREF('sim/flightmodel/position/P', 50) # setting roll rate to 50

        # setX = posx + 25
        # client.sendDREF(posxDREF, setX)
        # setZ = posz + 25
        # client.sendDREF(poszDREF, setZ)
        setY = posy + 10
        # client.sendDREF(posyDREF, setY)

        reward = 0
        # got_reward_from_alt = False
        while True:
        #     sleep(0.1)
        #     altiDREF = 'sim/cockpit2/gauges/indicators/radio_altimeter_height_ft_pilot'
        #     altitude = client.getDREF(altiDREF)[0] * 0.3048
        #     reward = reward - 20 - math.sqrt(abs(altitude - 22.2156))
        #     if altitude < 1 and not got_reward_from_alt:  # if the rocket hit the target altitude and position
        #         reward += 6000
        #         got_reward_from_alt = True
            # print("reward:", reward)

            # print(altitude)
            # get the wind info


            crash = client.getDREF("sim/flightmodel2/misc/has_crashed")
            if crash[0]:
                reward -= 100000
                print("crash")
                break

        return reward


if __name__ == "__main__":
    reward = launch()
    print("reward:", reward)
