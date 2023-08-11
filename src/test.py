# # import numpy as np
# # dt = 0.1
# # end_t = 20
# # t = np.arange(0, end_t+dt, dt)
# # t_lin = np.linspace(0, end_t, end_t*10+1)
# #
# # print(t)
# # print(len(t))
# # print(t_lin)
# # print(len(t_lin))
#
# class Test:
#     def __init__(self):
#         self.x = 20
#         print("self.x:", self.x)
#
#         self._reset()
#
#     def _reset(self):
#         self.y = 10
#         print("self.y:", self.y)
#
# test = Test()
# import sklearn.preprocessing
#
# normalizer = sklearn.preprocessing.StandardScaler()
# print("normalizer:", normalizer)

# from sklearn.preprocessing import StandardScaler
# data = [[0, 0], [0, 0], [1, 1], [1, 1]]
# scaler = StandardScaler()
# print(scaler.fit(data))
# print(scaler.mean_)
# from gym import spaces
# import numpy as np
# observation_space = spaces.Box(-np.inf, +np.inf, (6,))
# print(observation_space)

import xpc
import time
client = xpc.XPlaneConnect()

# Verify connection
try:
    # If X-Plane does not respond to the request, a timeout error
    # will be raised.
    client.getDREF("sim/test/test_float")


except Exception as e:
    print("Error establishing connection to X-Plane.")
    print("Exiting...")

# altitude = client.getPOSI()[2]
# print("initial altitude:", altitude)
X = -998
values = [X, X, 1000, X, X, 0, 1]
# client.sendPOSI(values, 0)
posxDREF = 'sim/flightmodel/position/local_x'
# posyDREF = 'sim/flightmodel/position/local_y'
# poszDREF = 'sim/flightmodel/position/local_z'
posx = client.getDREF(posxDREF)[0]
# posy = client.getDREF(posyDREF)[0]
# posz = client.getDREF(poszDREF)[0]
# setX = -15531 + 20
# client.sendDREF(posxDREF, setX)
# setZ = -55351 + 39
# client.sendDREF(poszDREF, setZ)

client.sendDREF('sim/flightmodel/position/true_airspeed', 10)
x = client.getDREF('sim/flightmodel/forces/fside_total')[0]
print(x)
client.sendDREF('sim/operation/override/override_forces', 0)
t = 1
while True:
    client.sendDREF('sim/operation/override/override_forces', 1)
    client.sendDREF('sim/flightmodel/forces/fside_total', 10000000)
    x = client.getDREF('sim/flightmodel/forces/fside_total')[0]
    client.sendDREF('sim/operation/override/override_forces', 0)
    print(x)
    time.sleep(1)
    t += 1
    if t == 5:
        break

#
#
# time.sleep(2)
#
# client.sendDREF('sim/operation/override/override_forces', 1)
#
# client.sendDREF('sim/flightmodel/forces/fside_total', -50000000)
# x = client.getDREF('sim/flightmodel/forces/fside_total')[0]
# print(x)
# client.sendDREF('sim/operation/override/override_forces', 0)
# setY = posy + 25
# client.sendDREF(posyDREF, setY)
# posxDREF = 'sim/flightmodel/position/local_x'
# posyDREF = 'sim/flightmodel/position/local_y'
# poszDREF = 'sim/flightmodel/position/local_z'
# posx = client.getDREF(posxDREF)[0]
# posy = client.getDREF(posyDREF)[0]
# posz = client.getDREF(poszDREF)[0]

# print("pos x:", posx, "pos y:", posy, "pos z:", posz)
#
# if (-2.75 * posx - 98154.25) <= posz <= (-2.75 * posx - 97968.25):
#     if (0.35 * posx - 49946.15) <= posz <= (0.35 * posx - 49884.15):
#         print("rocket is inside the lander")
#     else:
#         print("rocket is outside the lander")
# else:
#     print("rocket is outside the lander")



# while True:
#     print("alti:", client.getPOSI())

# sim/flightmodel/failures/onground_any	int	y	???	User Aircraft is on the ground when this is set to 1
client.close()

print("connection closed")
