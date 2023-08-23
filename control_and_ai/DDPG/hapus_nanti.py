import time
import xpc

client = xpc.XPlaneConnect()
values = [-998, -998, 200, -998, -998, -998, -998]
# has_crash = 0
client.sendPOSI(values, 0)
#
# while has_crash == 0:
#     has_crash = client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
#     print("has crashed:", has_crash)
#     time.sleep(1)
#
# while has_crash:
#     has_crash = client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
#     print("has crashed:", has_crash)
#     time.sleep(1)
#
# client.sendPOSI(values, 0)
#
# while has_crash == 0:
#     has_crash = client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
#     print("has crashed:", has_crash)
#     time.sleep(1)
#
# while has_crash:
#     has_crash = client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
#     print("has crashed:", has_crash)
#     time.sleep(1)
#
# client.sendPOSI(values, 0)
#
# while has_crash == 0:
#     has_crash = client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
#     print("has crashed:", has_crash)
#     time.sleep(1)
#
# while has_crash:
#     has_crash = client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
#     print("has crashed:", has_crash)
#     time.sleep(1)
for i in range(100):
    print(client.getDREF("sim/flightmodel/forces/vz_acf_axis"))
    print(client.getDREF("sim/flightmodel/position/vh_ind"))

client.close()


# sim/operation/reset_flight                         Reset flight to most recent start.
# sim/operation/go_to_default                        Reset flight to nearest airport.
# sim/operation/reset_to_runway                      Reset flight to nearest runway.
# sim/operation/go_next_runway                       Reset flight to next runway on current airport