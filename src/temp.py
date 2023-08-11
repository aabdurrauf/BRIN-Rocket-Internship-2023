def set_rocket(self, client):
    # set position (altitude)
    X = -998
    values = [X, X, 1000, X, X, X, X]
    client.sendPOSI(values, 0)

    # set fuel - no need because it automatically sets in X-Plane
    # fuelDREF = 'sim/flightmodel/weight/m_fuel1'
    # client.sendDREF(fuelDREF, INITIAL_FUEL_MASS)
