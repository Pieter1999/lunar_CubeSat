from astroquery.jplhorizons import Horizons


class Constants:
    """
    This class contains all the constants that are used throughout the research. Constants can in this way
    easily be called and their values are stored in one central location. Further in-line comments will
    clarify the meaning and values of the constants present.
    """

    # Physical constants
    G = 6.67428e-11  # Universal Gravitational Constant [m^3s^-2kg^-1]
    mu_Earth = 398600441800000.0  # Gravitational parameter Earth [m^3s^-2]
    mu_Moon = 4.9048695e12  # Gravitational parameter Moon [m^3s^-2]
    mu_Sun = 1.32712440018e20  # Gravitational parameter Sun [m^3s^-2]
    AU = 149597870700.0  # Astronomical unit [m]
    R_Earth = 6.3781e6  # Earth mean radius [m]
    R_Moon = 1737.4e3  # Moon mean radius [m]


class DisturbanceTorques:
    """
    This class contains the disturbance torques used for the ADCS modelling. These will be the gravity gradient torque and
    solar radiation pressure torque described from literature.
    """


class EphemerisData:
    """
    This class contains the functions for the retrieval of ephemeris data. This data is collected from the JPL Horizons module
    and can be retrieved in two separate ways. The function vectors retrieves Cartesian coordinates and velocities over time, whereas the
    function keplerian retrieves the Keplerian elements (semi-major axis, eccentricity, etc.) over time.
    """

    def __init__(self, t0, t1, id, location):
        self.t0 = t0
        self.t1 = t1
        self.id = id
        self.location = location

    def vectors(self):
        object = Horizons(id=self.id, location=self.location, epochs=[self.t0, self.t1])

        ephemeris = object.vectors()

        cartesian = ephemeris["datetime_jd", "x", "y", "z", "vx", "vy", "vz"]

        return cartesian

    def keplerian(self):
        object = Horizons(id=self.id, location=self.location, epochs=[self.t0, self.t1])

        ephemeris = object.elements()

        keplerian = ephemeris["datetime_jd", "a", "e", "incl", "Omega", "w", "nu"]

        return keplerian


eph = EphemerisData(2459600.5, 2459601.5, "499", "500@10")

lst = eph.keplerian()

print(lst["a"])
