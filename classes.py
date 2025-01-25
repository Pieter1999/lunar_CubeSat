from astroquery.jplhorizons import Horizons
from astropy.time import Time
from datetime import date
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import null_space
import pandas as pd
import os
import re
from scipy.optimize import minimize

class Constants:
    """
    This class contains all the constants that are used throughout the research. Constants can in this way
    easily be called and their values are stored in one central location. Further in-line comments will
    clarify the meaning and values of the constants present.
    """

    # Physical constants
    c = 299792458  # Speed of light [ms^-1]
    G = 6.67428e-11  # Universal Gravitational Constant [m^3s^-2kg^-1]
    mu_Earth = 398600441800000.0  # Gravitational parameter Earth [m^3s^-2]
    mu_Moon = 4.9048695e12  # Gravitational parameter Moon [m^3s^-2]
    mu_Sun = 1.32712440018e20  # Gravitational parameter Sun [m^3s^-2]
    AU = 149597870700.0  # Astronomical unit [m]
    R_Earth = 6.3781e6  # Earth mean radius [m]
    R_Moon = 1737.4e3  # Moon mean radius [m]
    R_Sun = 695700e3  # Sun mean radius [m]
    P_solar = 3.842e26  # Power exerted by the Sun, [W]

    # LUMIO spacecraft data
    inertia_matrix = np.array([[100.9, 0, 0], [0, 25.1, 0], [0, 0, 91.6]]) * 10 ** (
        -2
    )  # deployed inertia matrix, [kg m^2]
    inertia_matrix_undeployed = np.array(
        [[30.5, 0, 0], [0, 20.9, 0], [0, 0, 27.1]]
    ) * 10 ** (-2)

    # Payload panel location
    LUMIO_loc_pp = np.array([0, 0.15, 0])  # m

    # Hyperion RW400 data
    T_RW_max = 0.007 #Nm
    P_peak_max = 9 #W

    # Solid State Propulsion Pocket Rocket data
    F_SSP_max = 0.0002 #N
    P_SSP_max = 20 #W
    MIB = 1.18e-6 #Ns

    # Astroquery data
    # CAPSTONE data
    id_CAPSTONE = "-1176"  # JPL Horizons
    start_date_CAPSTONE = (
        "2022-11-14"  # Official nominal mission start date, insertion in NRHO
    )
    end_date_CAPSTONE = "2023-05-18"  # Official end date nominal mission

    # Location data
    id_Moon = "301"
    id_Sun = "10"
    id_Earth = "500"
    location_Moon_centre = "500@301"  # 500 indicates the body-centric location
    location_Sun_centre = "500@10"
    location_Earth_centre = "500"
    location_CAPSTONE_centre = "500@-1176"


class Rotation:
    """
    Placeholdere name. This class will hold several rotational dynamics functions for easy use in reference frame conversion practices.
    At the moment of writing, only the 3-2-1 Euler rotation method and Euler angle - quaternion conversions are considered for this class.
    Take three Euler angles as input, potentially angular velocity as well. NOTE: roll = theta_1, pitch = theta_2, yaw = theta_3
    """

    def euler_rotation(
        self, theta_1, theta_2, theta_3
    ):  # 3-2-1 rotation, using Euler angles, from INERTIAL TO BODY FRAME, is R_x @ R_y @ R_z. Other way around is the inverse of this.
        # Define rotation matrices
        R_z = np.array(
            [
                [np.cos(theta_3), np.sin(theta_3), 0],
                [-np.sin(theta_3), np.cos(theta_3), 0],
                [0, 0, 1],
            ]
        )
        R_y = np.array(
            [
                [np.cos(theta_2), 0, -np.sin(theta_2)],
                [0, 1, 0],
                [np.sin(theta_2), 0, np.cos(theta_2)],
            ]
        )
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta_1), np.sin(theta_1)],
                [0, -np.sin(theta_1), np.cos(theta_1)],
            ]
        )

        return R_x @ R_y @ R_z

    def quaternion_321_rotation(
        self, qw, qx, qy, qz
    ):  # 3-2-1 rotation, using quaternions, is the inverse of the resultant matrix, so np.linalg.inv(R) should be used to match Euler 3-2-1 rotation from inertial to body frame.
        # This regular matrix is other way around.

        R = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ]
        )

        return np.linalg.inv(R)

    def euler_to_quaternion(self, theta_1, theta_2, theta_3):

        # Extract individual angles
        cr, sr = np.cos(theta_1 * 0.5), np.sin(theta_1 * 0.5)
        cp, sp = np.cos(theta_2 * 0.5), np.sin(theta_2 * 0.5)
        cy, sy = np.cos(theta_3 * 0.5), np.sin(theta_3 * 0.5)

        # Calculate quaternion components, based on the 3-2-1 Euler sequence
        qw = cr * cp * cy + sr * sp * sy  # Real component
        qx = sr * cp * cy - cr * sp * sy  # Imaginary, x
        qy = cr * sp * cy + sr * cp * sy  # Imaginary, y
        qz = cr * cp * sy - sr * sp * cy  # Imaginary, z

        return np.array([qw, qx, qy, qz])  # Sequence w, x, y, z

    def quaternion_product(self, q1, q2):  # Kronecker definition
        """
        Compute the product of two quaternions.
        Each quaternion is represented as an array [qw, qx, qy, qz]

        Args:
        q1 (array): The first quaternion as (w, x, y, z).
        q2 (array): The second quaternion as (w, x, y, z).

        Returns:
        array: The resulting quaternion product as (w, x, y, z).
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        # Calculate the product components
        qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([qw, qx, qy, qz])  # Sequence w, x, y, z

    def quaternion_to_euler(
        self, qw, qx, qy, qz
    ):  # Verified, using euler_to_quat first, then quat_to_euler

        # Compute Euler angles
        t0 = 2.0 * (qw * qx + qy * qz)
        t1 = 1.0 - 2.0 * (qx**2 + qy**2)
        roll = np.arctan2(t0, t1)

        t2 = 2.0 * (qw * qy - qz * qx)
        t2 = np.clip(t2, -1.0, 1.0)  # Clamp to avoid numerical errors on the boundaries
        pitch = np.arcsin(t2)

        t3 = 2.0 * (qw * qz + qx * qy)
        t4 = 1.0 - 2.0 * (qy**2 + qz**2)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw])
    
    # Below definition of DCM to quaternion conversion is incorrect; it assumes a DCM consisting of quaternions
    # This raises instability. Below code can be verified with the PyQuaternion module.
    # New DCM_to_quaternion function settles this
    def DCM_to_quaternion_old(self, DCM_matrix):
        a11 = DCM_matrix[0,0]
        a22 = DCM_matrix[1,1]
        a33 = DCM_matrix[2,2]

        qw = 1/2 * np.sqrt(1 + a11 + a22 + a33)
        q1 = 1/2 * np.sqrt(1 + a11 - a22 - a33)
        q2 = 1/2 * np.sqrt(1 - a11 + a22 - a33)
        q3 = 1/2 * np.sqrt(1 - a11 - a22 + a33)

        return np.array([qw, q1, q2, q3])

    def DCM_to_quaternion(self, DCM_matrix):
        a11 = DCM_matrix[0, 0]
        a22 = DCM_matrix[1, 1]
        a33 = DCM_matrix[2, 2]
        trace = a11 + a22 + a33

        if trace > 0:
            qw = 0.5 * np.sqrt(1 + trace)
            qx = (DCM_matrix[2, 1] - DCM_matrix[1, 2]) / (4 * qw)
            qy = (DCM_matrix[0, 2] - DCM_matrix[2, 0]) / (4 * qw)
            qz = (DCM_matrix[1, 0] - DCM_matrix[0, 1]) / (4 * qw)
        elif a11 > a22 and a11 > a33:
            qx = 0.5 * np.sqrt(1 + a11 - a22 - a33)
            qw = (DCM_matrix[2, 1] - DCM_matrix[1, 2]) / (4 * qx)
            qy = (DCM_matrix[0, 1] + DCM_matrix[1, 0]) / (4 * qx)
            qz = (DCM_matrix[0, 2] + DCM_matrix[2, 0]) / (4 * qx)
        elif a22 > a33:
            qy = 0.5 * np.sqrt(1 + a22 - a11 - a33)
            qw = (DCM_matrix[0, 2] - DCM_matrix[2, 0]) / (4 * qy)
            qx = (DCM_matrix[0, 1] + DCM_matrix[1, 0]) / (4 * qy)
            qz = (DCM_matrix[1, 2] + DCM_matrix[2, 1]) / (4 * qy)
        else:
            qz = 0.5 * np.sqrt(1 + a33 - a11 - a22)
            qw = (DCM_matrix[1, 0] - DCM_matrix[0, 1]) / (4 * qz)
            qx = (DCM_matrix[0, 2] + DCM_matrix[2, 0]) / (4 * qz)
            qy = (DCM_matrix[1, 2] + DCM_matrix[2, 1]) / (4 * qz)

        return np.array([qw, qx, qy, qz])



class DisturbanceTorques:
    """
    This class contains the disturbance torques used for the ADCS modelling. These will be the gravity gradient torque and
    solar radiation pressure torque described from literature. The Gravity Gradient torque function returns a torque vector...
    """

    def __init__(self, inertia_matrix):

        self.inertia_matrix = inertia_matrix

    def GGMoon(self, q, position_SC_Moon):

        # Generate the rotation matrix to express the position vector in the body frame instead of the inertial Moon-centered frame (retrieve wrt this frame)
        rot = Rotation()
        rot_matrix = rot.quaternion_321_rotation(q[0], q[1], q[2], q[3])

        # Generate the unit vector of the spacecraft position vector in the body frame
        # First approximation -> needs to be verified, in the body frame as well
        R_sc = rot_matrix @ position_SC_Moon  # 3x1 vector
        R_sc_hat = R_sc / np.linalg.norm(R_sc)

        # Calculate gravity gradient torque in the body frame
        T_GG = (
            3
            * Constants.mu_Moon
            / (np.linalg.norm(R_sc) ** 3)
            * (np.cross(R_sc_hat, np.dot(self.inertia_matrix, R_sc_hat)))
        )

        return T_GG

    # DEFAULT: undeployed scenario. For deployed scenario, see next definition
    def SRP(self, q,  position_Sun_Moon, position_SC_Moon): 
        
        rot = Rotation()

        rho_s = 0.6  # These values have been taken from the LUMIO ADCS paper for now; to be verified or adjusted
        rho_d = 0.1

        # First, define surface CoM locations wrt satellite CoM [0,0,0] in body frame
        # Numbering according to ADCS LUMIO paper
        # Different reference frame, same as thruster config. Surface 1 is in positive z-axis, surface 4 is positive x-axis, surface 6 is positive y-axis
        # All values in meter, sides are either 20cm or 30cm (locations 5 and 6 are on the far sides)

        S_loc_1 = np.array([0, 0, 0.1])
        S_loc_2 = np.array([0, 0, -0.1])
        S_loc_3 = np.array([-0.1, 0, 0])
        S_loc_4 = np.array([0.1, 0, 0])
        S_loc_5 = np.array([0, -0.15, 0])
        S_loc_6 = np.array([0, 0.15, 0])

        c_p = np.column_stack(
            [S_loc_1, S_loc_2, S_loc_3, S_loc_4, S_loc_5, S_loc_6]
        )  # Centre of pressure locations for calculation
        n_s = np.array([[0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1], [-1, 1, 0, 0, 0, 0]])
        r_S_SC = position_Sun_Moon - position_SC_Moon
        S_inertial = np.column_stack(
            [
                (S_loc_1 - r_S_SC) / np.linalg.norm(S_loc_1 - r_S_SC),
                (S_loc_2 - r_S_SC) / np.linalg.norm(S_loc_2 - r_S_SC),
                (S_loc_3 - r_S_SC) / np.linalg.norm(S_loc_3 - r_S_SC),
                (S_loc_4 - r_S_SC) / np.linalg.norm(S_loc_4 - r_S_SC),
                (S_loc_5 - r_S_SC) / np.linalg.norm(S_loc_5 - r_S_SC),
                (S_loc_6 - r_S_SC) / np.linalg.norm(S_loc_6 - r_S_SC),
            ]
        )

        # Rotation matrix to body frame
        rot_matrix = rot.quaternion_321_rotation(q[0], q[1], q[2], q[3])
        # Convert S-matrix to the body frame
        S = rot_matrix @ S_inertial

        A = np.array([6, 6, 6, 6, 4, 4]) * 10 ** (
            -2
        )  # Surface areas of all labeled surfaces, in m2
        F = np.empty((3, len(A)))

        # Calculate the solar intensity at the spacecraft location, as seen in LUMIO ADCS paper

        I = Constants.P_solar / (4 * np.pi * np.linalg.norm(r_S_SC) ** 2)

        for i in range(len(A)):
            F_SRP = (
                I
                / Constants.c
                * A[i]
                * np.dot(S.T[i], n_s.T[i])
                * (
                    (1 - rho_s) * S.T[i]
                    + (2 * rho_s * np.dot(S.T[i], n_s.T[i]) + 2 / 3 * rho_d) * n_s.T[i]
                )
            ) 

            if np.dot(S.T[i], n_s.T[i]) > 0:
                F[0, i], F[1, i], F[2, i] = np.cross(c_p[:, i], F_SRP)

            else:
                F[:, i] = np.zeros(3)

        return F.sum(axis=1)
    
    def SRP_deployed(self, q, position_Sun_Moon, position_SC_Moon):

        rot = Rotation()

        rho_s = 0.6  # These values have been taken from the LUMIO ADCS paper for now; to be verified or adjusted
        rho_d = 0.1

        # First, define surface CoM locations wrt satellite CoM [0,0,0] in body frame
        # Numbering according to ADCS LUMIO paper
        # Different reference frame, same as thruster config. Surface 1 is in positive z-axis, surface 4 is positive x-axis, surface 6 is positive y-axis
        # All values in meter, sides are either 20cm or 30cm (locations 5 and 6 are on the far sides)
        # Solar arrays lie 

        S_loc_1 = np.array([0, 0, 0.1])
        S_loc_2 = np.array([0, 0, -0.1])
        S_loc_3 = np.array([-0.1, 0, 0])
        S_loc_4 = np.array([0.1, 0, 0])
        S_loc_5 = np.array([0, -0.15, 0])
        S_loc_6 = np.array([0, 0.15, 0])
        S_loc_7 = np.array([-0.45, 0, 0])
        S_loc_8 = np.array([-0.45, 0, 0])
        S_loc_9 = np.array([0.45, 0, 0])
        S_loc_10 = np.array([0.45, 0, 0])

        c_p = np.column_stack(
            [S_loc_1, S_loc_2, S_loc_3, S_loc_4, S_loc_5, S_loc_6, S_loc_7, S_loc_8, S_loc_9, S_loc_10]
        ) # Centre of pressure locations for calculation

        r_S_SC = position_Sun_Moon - position_SC_Moon

        S_inertial = np.column_stack(
            [
                (S_loc_1 - r_S_SC) / np.linalg.norm(S_loc_1 - r_S_SC),
                (S_loc_2 - r_S_SC) / np.linalg.norm(S_loc_2 - r_S_SC),
                (S_loc_3 - r_S_SC) / np.linalg.norm(S_loc_3 - r_S_SC),
                (S_loc_4 - r_S_SC) / np.linalg.norm(S_loc_4 - r_S_SC),
                (S_loc_5 - r_S_SC) / np.linalg.norm(S_loc_5 - r_S_SC),
                (S_loc_6 - r_S_SC) / np.linalg.norm(S_loc_6 - r_S_SC),
                (S_loc_7 - r_S_SC) / np.linalg.norm(S_loc_7 - r_S_SC),
                
                (S_loc_8 - r_S_SC) / np.linalg.norm(S_loc_8 - r_S_SC),
                (S_loc_9 - r_S_SC) / np.linalg.norm(S_loc_9 - r_S_SC),
                (S_loc_10 - r_S_SC) / np.linalg.norm(S_loc_10 - r_S_SC),
            ]
        )

        # Rotation matrix to body frame
        rot_matrix = rot.quaternion_321_rotation(q[0], q[1], q[2], q[3])
        # Convert S-matrix to the body frame
        S = rot_matrix @ S_inertial

        A = np.array([6, 6, 6, 6, 4, 4, 12, 12, 12, 12]) * 10 ** (
            -2
        )  # Surface areas of all labeled surfaces, in m2

        # Define the n_s array as a function of alpha
        def normal_s(alpha):
            return np.array([
                [0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, np.sin(alpha), -np.sin(alpha), np.sin(alpha), -np.sin(alpha)],
                [-1, 1, 0, 0, 0, 0, np.cos(alpha), -np.cos(alpha), np.cos(alpha), -np.cos(alpha)]
            ])

        # Define the objective function to minimize (negative total dot product to maximize it)
        def objective(alpha_array):
            alpha = alpha_array[0]
            n_s_alpha = normal_s(alpha)  # Get n_s for this alpha
            total_dot_product = np.sum([np.dot(n_s_alpha[:, i], S[:, i]) for i in range(10)])
            return -total_dot_product  # Minimize the negative to maximize the positive

        # Use scipy.optimize to find the alpha that maximizes the total dot product
        result = minimize(objective, x0=0)  # x0 is the initial guess for alpha
        alpha_optimal = result.x[0]  # Optimal alpha value

        n_s = normal_s(alpha_optimal)

        F = np.empty((3, len(A)))

        # Calculate the solar intensity at the spacecraft location, as seen in LUMIO ADCS paper

        I = Constants.P_solar / (4 * np.pi * np.linalg.norm(r_S_SC) ** 2)

        for i in range(len(A)):
            F_SRP = (
                I
                / Constants.c
                * A[i]
                * np.dot(S.T[i], n_s.T[i])
                * (
                    (1 - rho_s) * S.T[i]
                    + (2 * rho_s * np.dot(S.T[i], n_s.T[i]) + 2 / 3 * rho_d) * n_s.T[i]
                )
            ) 

            if np.dot(S.T[i], n_s.T[i]) > 0:
                F[0, i], F[1, i], F[2, i] = np.cross(c_p[:, i], F_SRP)

            else:
                F[:, i] = np.zeros(3)

        return F.sum(axis=1)

class EphemerisData:
    """
    This class contains the functions for the retrieval of ephemeris data. This data is collected from the JPL Horizons module
    and can be retrieved in two separate ways. The function "vectors" retrieves Cartesian coordinates and velocities over time, whereas the
    function "keplerian" retrieves the Keplerian elements (semi-major axis, eccentricity, etc.) over time.

    Input:
    id: string, for Horizons query
    location: string, for Horizons query
    time_step: number, hours desired for time step of epoch query. Fractionals possible to indicate minutes / seconds
    start_date: custom start date, default set to beginning of nominal CAPSTONE mission, convention: YEAR-MONTH-DAY HOUR:MINUTE:SECOND
    end_date: custom end date, default set to end of nominal CAPSTONE mission
    """

    def __init__(self, id, location, time_step, start_date=None, end_date=None):

        # If no input start or end dates are given, input the CAPSTONE mission start and end dates

        if start_date is None:
            start_date = Constants.start_date_CAPSTONE
        if end_date is None:
            end_date = Constants.end_date_CAPSTONE

        # Validate input start and end date, to verify they fall within the nominal mission duration, see definition
        self.validate_dates(start_date, end_date)

        # Convert scalar dates (input) to Julian dates for Horizons query input
        self.t0 = start_date
        self.t1 = end_date
        self.id = id
        self.location = location
        self.time_step = f"{int(time_step * 60)}m"  # Convert fractional time_step to minutes, input for class remains hours

    # Validation of correct date usage, simple if-statement for range description, in jd
    def validate_dates(self, start, end):
        mission_start = Time(
            Constants.start_date_CAPSTONE, format="iso", scale="utc"
        ).jd
        mission_end = Time(Constants.end_date_CAPSTONE, format="iso", scale="utc").jd
        start_jd = Time(start, format="iso", scale="utc").jd
        end_jd = Time(end, format="iso", scale="utc").jd

        if not (mission_start <= start_jd <= end_jd <= mission_end):
            raise ValueError(
                "Provided dates must be within the official mission dates."
            )

    def vectors(self):
        # Retrieve desired object
        object = Horizons(
            id=self.id,
            location=self.location,
            epochs={"start": self.t0, "stop": self.t1, "step": self.time_step},
        )

        # Query Cartesian coordinate vectors, full table
        ephemeris = object.vectors()

        # Query specific columns from list
        cartesian = ephemeris["datetime_jd", "x", "y", "z", "vx", "vy", "vz"]

        # Data storage process
        if self.id == "301":
            body = "Moon"
        elif self.id == "-1176":
            body = "CAPSTONE"
        elif self.id == "10":
            body = "Sun"
        elif self.id == "500":
            body = "Earth"
        else:
            body = "other"

        if self.location == "500@301":
            center = "Moon-centered"
        elif self.location == "500@10":
            center = "Sun-centered"
        elif self.location == "500":
            center = "Earth-centered"
        elif self.location == "500@-1176":
            center = "CAPSTONE-centered"

        name = (
            "/Users/pieter/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Research Phase/lunar_CubeSat/ephemeris_data/cartesian_coordinates_"
            + body
            + "_"
            + center
            + "_"
            + self.t0
            + "_to_"
            + self.t1
            + "_"
            + str(self.time_step)
            + ".dat"
        )

        np.savetxt(
            name,
            cartesian,
        )

        return name, cartesian

    def keplerian(self):
        object = Horizons(
            id=self.id,
            location=self.location,
            epochs={"start": self.t0, "stop": self.t1, "step": self.time_step},
        )

        # Query complete elements table
        ephemeris = object.elements()

        # Query desired Keplerian elements
        keplerian = ephemeris["datetime_jd", "a", "e", "incl", "Omega", "w", "nu"]

        # Data storage process
        if self.id == "301":
            body = "Moon"
        elif self.id == "-1176":
            body = "CAPSTONE"
        elif self.id == "10":
            body = "Sun"
        elif self.id == "500":
            body = "Earth"
        else:
            body = "other"

        if self.location == "500@301":
            center = "Moon-centered"
        elif self.location == "500@10":
            center = "Sun-centered"
        elif self.location == "500":
            center = "Earth-centered"
        elif self.location == "500@-1176":
            center = "CAPSTONE-centered"

        np.savetxt(
            "/Users/pieter/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Research Phase/lunar_CubeSat/ephemeris_data/keplerian_elements_"
            + body
            + "_"
            + center
            + "_"
            + self.t0
            + "_to_"
            + self.t1
            + "_"
            + str(self.time_step)
            + ".dat",
            keplerian,
        )

        return keplerian

    def convert_data(self, data_file, control_time_step):

        original_data = np.loadtxt(data_file)

        # Determine time step from original data
        original_time_step = round(
            (original_data[1, 0] - original_data[0, 0]) * 24 * 3600
        )

        # Convert first original data column to seconds since epoch
        for i in range(len(original_data[:, 0])):
            original_data[i, 0] = i * original_time_step
            original_data[i, 1:4] *= Constants.AU
            original_data[i, 4:] *= Constants.AU / 24 / 3600

        # From this point onwards, data becomes converted to the desired number of time steps for the control algorithm
        min_number_intervals = int(original_time_step / control_time_step)

        converted_data = np.repeat(original_data, min_number_intervals, axis=0)

        time_list = []

        for j in range(len(original_data[:, 0])):
            for k in range(min_number_intervals):
                new_time = (
                    converted_data[j * min_number_intervals, 0] + control_time_step * k
                )
                time_list.append(new_time)

        converted_data[:, 0] = time_list

        # Save using adjusted naming in different repository
        base_name = os.path.basename(data_file)
        np.savetxt(
            "/Users/pieter/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Research Phase/lunar_CubeSat/converted_ephemeris_data/"
            + "converted_"
            + str(control_time_step)
            + "s_"
            + base_name,
            converted_data,
        )

    def convert_data_interpolated(self, data_file, control_time_step):
    
        original_data = np.loadtxt(data_file)

        # Determine time step from original data
        original_time_step = round(
            (original_data[1, 0] - original_data[0, 0]) * 24 * 3600
        )

        # Convert first original data column to seconds since epoch
        for i in range(len(original_data[:, 0])):
            original_data[i, 0] = i * original_time_step
            original_data[i, 1:4] *= Constants.AU
            original_data[i, 4:] *= Constants.AU / 24 / 3600

        # Prepare arrays for the new data set
        new_time_steps = np.arange(original_data[0, 0], original_data[-1, 0] + control_time_step, control_time_step)
        converted_data = np.zeros((len(new_time_steps), original_data.shape[1]))

        # Interpolate each column
        for col in range(1, original_data.shape[1]):  # Skip time column for interpolation
            converted_data[:, col] = np.interp(new_time_steps, original_data[:, 0], original_data[:, col])

        # Fill in the new time column
        converted_data[:, 0] = new_time_steps

        # Save using adjusted naming in different repository
        base_name = os.path.basename(data_file)
        np.savetxt(
            "/Users/pieter/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Research Phase/lunar_CubeSat/converted_ephemeris_data/"
            + "converted_"
            + str(control_time_step)
            + "s_"
            + base_name,
            converted_data,
        )


class PID:
    """
    Functions:
    """

    def __init__(
        self,
        inertia_matrix,
    ):
        """
        Initialize the PID controller with the required inertia matrix.

        Args:
            inertia_matrix (array): A numpy array representing the inertia matrix of the system.
        """
        self.I = inertia_matrix
        self.integral_term = np.zeros(3)  # Integral term is initiated at zero

    def quaternion_error(self, quaternion_vector, quaternion_ref_vector):
        """
        Compute the error between two quaternions.
        Each quaternion is represented as an array np.array([qw, qx, qy, qz]).

        Args:
        quaternion_vector (array): The actual quaternion vector as np.array([qw, qx, qy, qz])
        quaternion_ref_vector (array): The reference quaternion vector as np.array([qw_ref, qx_ref, qy_ref, qz_ref]).

        Returns:
        array: quaternion error vector
        """

        qrw, qr1, qr2, qr3 = quaternion_ref_vector
        qw, q1, q2, q3 = quaternion_vector

        adj_quaternion_vector = np.array([-q1, -q2, -q3, qw])
        adj_matrix = np.array([[qrw, qr3, -qr2, qr1], [-qr3, qrw, qr1, qr2], [qr2, -qr1, qrw, qr3], [-qr1, -qr2, qr3, qrw]])

        qe1, qe2, qe3, qew = adj_matrix @ adj_quaternion_vector

        return np.array([qew, qe1, qe2, qe3])

    def derivative_omega(self, omega_vector, T_d, T_c):
        """
        Calculate the derivative of the angular velocity vector (omega_dot) based on the current state and external torques.

        Args:
            T_d (array): Disturbance torque vector.
            T_c (array): Control torque vector.
            omega_vector (array): Current angular velocity vector.

        Returns:
            array: The derivative of the angular velocity (angular acceleration).
        """

        omega_dot_vector = np.linalg.inv(self.I) @ (
            T_c + T_d - np.cross(omega_vector, (self.I @ omega_vector))
        )

        return omega_dot_vector
    
    def derivative_omega_reactionwheel(self, omega_vector, T_d, T_c, h_rw):
        """
        Specifically for reaction wheel analysis; includes h_rw term in the equations of motion
        """

        omega_dot_vector = np.linalg.inv(self.I) @ (
            T_c + T_d - np.cross(omega_vector, (self.I @ omega_vector + h_rw))
        )

        return omega_dot_vector


    def derivative_quaternion(self, quaternion_vector, omega_vector):
        """
        Compute the time derivative of a quaternion based on the current angular velocity.

        Args:
            quaternion_vector (array): The current quaternion vector as np.array([qw, qx, qy, qz]).
            omega_vector (array): The angular velocity vector.

        Returns:
            array: The time derivative of the quaternion.
        """

        # Again, order qw qx qy qz
        omega_q_vector = np.append(0, omega_vector)
        rot = Rotation()

        q_dot = 0.5 * rot.quaternion_product(quaternion_vector, omega_q_vector)

        return q_dot

    def derivative_quaternion_error(
        self, quaternion_vector, quaternion_ref_vector, omega_vector
    ):  # See notes in notebook
        """
        Calculate the derivative of the quaternion error between the reference and current quaternion.

        Args:
            quaternion_vector (array): The current quaternion vector as np.array([qw, qx, qy, qz]).
            quaternion_ref_vector (array): The reference quaternion vector as np.array([qw, qx, qy, qz]).
            omega_vector (array): The angular velocity vector.

        Returns:
            array: The derivative of the quaternion error.
        """

        qcw, qcx, qcy, qcz = quaternion_ref_vector
        q_dot = self.derivative_quaternion(quaternion_vector, omega_vector)

        ref_matrix = np.array(
            [
                [qcw, qcx, qcy, qcz],
                [qcx, -qcw, qcz, -qcy],
                [qcy, -qcz, -qcw, qcx],
                [qcz, qcy, -qcx, -qcw],
            ]
        )

        return ref_matrix @ q_dot

    def control_torque(
        self, quaternion_vector, quaternion_ref_vector, omega_vector, k_p, k_i, k_d, k_s, dt
    ):
        """
        Compute the control torque based on PID control laws using the quaternion error and its derivative.

        Args:
            quaternion_vector (array): The current quaternion vector.
            quaternion_ref_vector (array): The reference quaternion vector.
            omega_vector (array): The angular velocity vector.
            k_p (array): Proportional gain coefficients (k_p_x, k_p_y, k_p_z).
            k_i (array): Integral gain coefficients (k_i_x, k_i_y, k_i_z).
            k_d (array): Derivative gain coefficients (k_d_x, k_d_y, k_d_z).
            dt (float): Time step for the integral calculation.

        Returns:
            array: Control torque vector [T_c_x, T_c_y, T_c_z].
        """
        qew, qex, qey, qez = self.quaternion_error(
            quaternion_vector, quaternion_ref_vector
        )  # quaternion error components

        omega_x, omega_y, omega_z = omega_vector

        k_p_1, k_p_2, k_p_3 = k_p  # proportional gain
        k_i_1, k_i_2, k_i_3 = k_i  # integral gain
        k_d_1, k_d_2, k_d_3 = k_d  # derivative gain
        k_s_1, k_s_2, k_s_3 = k_s  # speed gain


        int_x, int_y, int_z = self.integral_term

        # control torque in the x, y and z directions
        T_c_x = k_s_1 * (
            k_p_1 * qex + k_i_1 * int_x - k_d_1 * (omega_x)
        )
        T_c_y = k_s_2 * (k_p_2 * qey + k_i_2 * int_y - k_d_2 * (omega_y))
        T_c_z = k_s_3 * (k_p_3 * qez + k_i_3 * int_z - k_d_3 * (omega_z))

        self.integral_term += np.array([qex, qey, qez]) * dt

        return np.array([T_c_x, T_c_y, T_c_z])

    def rk4_integrator(self, func, y, dt, *args):
        """
        General RK4 integrator.
        Args:
            func: The function to calculate derivatives, signature func(y, *args)
            y: Current state variable (np.array)
            dt: Time step (float)
            *args: Additional arguments required by `func`
        Returns:
            np.array: Updated state after dt
        """
        k1 = func(y, *args) * dt
        k2 = func(y + 0.5 * k1, *args) * dt
        k3 = func(y + 0.5 * k2, *args) * dt
        k4 = func(y + k3, *args) * dt
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def euler_integrator(self, func, y, dt, *args):
        """
        Basic Euler integrator.
        Args:
            func: The function to calculate derivatives, with signature func(y, *args)
            y: Current state variable (np.array)
            dt: Time step (float)
            *args: Additional arguments required by `func`
        Returns:
            np.array: Updated state after dt
        """
        dy = func(y, *args)  # Calculate the derivative
        return y + dy * dt  # Update state
    
    def reference_omega(self, q_ref, omega_vector):
        q_ref_dot = self.derivative_quaternion(q_ref, omega_vector)
        q_ref_dot_adj = np.array([q_ref_dot[1], q_ref_dot[2], q_ref_dot[3], q_ref_dot[0]])

        mat = np.array([[q_ref[0], q_ref[3], -q_ref[2], -q_ref[0]], [q_ref[0], q_ref[3], -q_ref[2], -q_ref[0]], [q_ref[0], q_ref[3], -q_ref[2], -q_ref[0]]])

        return 2 * mat @ q_ref_dot_adj

    def reference_quaternion(self, position_SC_Moon, quaternion_vector):
        """
        This function takes the spacecraft position vector with respect to the Moon and the current spacecraft
        attitude defined in quaternions to calculate the reference quaternion at one instance of time. The
        reference is defined to have the upper panel (y+ direction) of the LUMIO spacecraft pointed towards
        the Moon and this should at all times be adhered to. The output is a vector [qw, qx, qy, qz] with the
        reference quaternion vectors to be used within the PID algorithm for error calculation.
        """

        qw, qx, qy, qz = quaternion_vector

        rot = Rotation()
        direction_SC_Moon = (
            -position_SC_Moon
        )  # Define the direction vector from the spacecraft to the Moon, inertial (Moon-centered) frame
        R = rot.quaternion_321_rotation(
            qw, qx, qy, qz
        )  # Rotation matrix for conversion from inertial to body frame
        direction_SC_Moon_body = np.dot(
            R, direction_SC_Moon
        )  # Convert to the body frame
        panel_body = Constants.LUMIO_loc_pp  # Retrieve desired panel CoM location

        n_current = panel_body / np.linalg.norm(
            panel_body
        )  # compare current orientation in the body frame to desired orientation in the body frame
        n_desired = direction_SC_Moon_body / np.linalg.norm(direction_SC_Moon_body)
        q_rot = np.array(
            [
                np.sqrt(np.linalg.norm(n_current) ** 2 * np.linalg.norm(n_desired) ** 2)
                + np.dot(n_current, n_desired)
            ]
            + list(np.cross(n_current, n_desired))
        )  # Rotation quaternion so that the panel is oriented towards the Moon
        q_rot /= np.linalg.norm(q_rot)  # Normalize the quaternion

        # Calculate the reference quaternion to adhere to, combination of current quaternion (rotation representation) and newly calculated quaternion q_rot
        return rot.quaternion_product(q_rot, quaternion_vector)
    

    def reference_quaternion_paper(self, position_SC_Moon, position_Sun_Moon):
        """
        Based on the paper "ATTITUDE CONTROL FOR THE LUMIO CUBESAT IN DEEP SPACE", outputting
        the desired attitude for the LUMIO spacecraft based on power maximisation. It can be 
        shown that this reference frame maximizes power generation by allowing the solar arrays 
        to be always normal to the Sun vector.

        ADJUSTMENT: in this case, considering the usage of a different reference frame
        , the negative y-plane should always be pointed towards the Moon, 
        """
        rot = Rotation()

        Sun_pointing_vector = (position_Sun_Moon - position_SC_Moon) / np.linalg.norm(position_Sun_Moon - position_SC_Moon)
        Moon_pointing_vector = - position_SC_Moon / np.linalg.norm(position_SC_Moon)

        x1 = Moon_pointing_vector
        x2 = np.cross(Sun_pointing_vector, x1) / np.linalg.norm(np.cross(Sun_pointing_vector, x1))
        x3 = np.cross(x1, x2) / np.linalg.norm(np.cross(x1, x2))

        A_d = np.column_stack((x2, -x1, x3))

        return rot.DCM_to_quaternion(A_d)


class Visualization:
    """
    Visualization class for result from attitude control.
    """

    def __init__(self, time_array):
        self.t = time_array

    def extract_dates_from_filename(self, filename):
        # Define the regular expression pattern to match the dates
        pattern = r'_(\d{4}-\d{2}-\d{2} \d{2}:\d{2})_to_(\d{4}-\d{2}-\d{2} \d{2}:\d{2})_'
        
        # Search the pattern in the filename
        match = re.search(pattern, filename)
        
        if match:
            # Extract the start and end date strings
            start_date_str = match.group(1)
            end_date_str = match.group(2)
            
            # Convert the strings to datetime objects
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M')
            
            return start_date, end_date
        else:
            raise ValueError("The filename does not match the expected pattern")

    def plot_trajectories(self, *data_files):
        plt.style.use('fast')
        fig = plt.figure(figsize=(10, 10))  # Set the figure size to 10x10
        ax = fig.add_subplot(111, projection="3d")

        # Define reference frames with adjusted Moon radius for a smaller sphere
        reference_frames = {
            "Sun": Constants.R_Sun,
            "Moon": Constants.R_Moon * 0.5,  # Reduced radius for Moon
            "Earth": Constants.R_Earth,
            "CAPSTONE": 0.2,
        }  # Radii in meters
        used_centers = set()

        first_file = True
        all_epochs = None

        for data_file in data_files:
            # Extracting information from file name
            parts = data_file.split("_")
            body_name = parts[6]
            center_part = parts[7]
            center = center_part.split("-")[0]

            print(center)

            # Check if center has been used consistently
            if used_centers and center not in used_centers:
                raise ValueError(
                    f"Inconsistent reference systems: {used_centers.pop()} vs {center}. All data must be in the same reference frame."
                )
            used_centers.add(center)

            # Reading data
            data = np.loadtxt(data_file)
            epochs, x, y, z, vx, vy, vz = data.T
            x, y, z = x, y, z  # AU to meters

            # Plotting trajectories
            ax.plot(x, y, z, label=body_name)

            # Check epochs
            if all_epochs is None:
                all_epochs = epochs
            elif not np.array_equal(all_epochs, epochs):
                raise ValueError(
                    "Epochs do not match across data files. Ensure all input files cover the same time periods."
                )

            # Centering reference body
            if center in reference_frames:
                # Plotting central body as a sphere
                radius = reference_frames[center]
                u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
                sphere_x = radius * np.cos(u) * np.sin(v)
                sphere_y = radius * np.sin(u) * np.sin(v)
                sphere_z = radius * np.cos(v)
                ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="grey", alpha=0.5)
                ax.text(0, 0, -5 * radius, center, color="black", fontsize=16, ha="center")

        # Setting limits for equal aspect ratio
        max_radius = max(
            abs(ax.get_xlim()[0]), abs(ax.get_ylim()[0]), abs(ax.get_zlim()[0])
        )
        ax.set_xlim([-max_radius, max_radius])
        ax.set_ylim([-max_radius, max_radius])
        ax.set_zlim([-max_radius, max_radius])

        # Add labels, title, and legend with adjusted font sizes
        ax.set_xlabel("X (m)", fontsize=16)
        ax.set_ylabel("Y (m)", fontsize=16)
        ax.set_zlabel("Z (m)", fontsize=16)
        ax.legend(fontsize=16)

        # Adjust tick parameters for axis font sizes
        ax.tick_params(axis='both', which='major', labelsize=16)

        plt.show()


        # # Save plot
        # today = date.today()
        # plt.savefig("/Users/pieter/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Research Phase/lunar_CubeSat/figures/trajectory_rundate_" + str(today) + "_length_" + str(round(float(len(self.t)) / 60 / 60 / 24, 3)) + "_days_timestep_" + str(self.t[1] - self.t[0]) + "_seconds.png")

    # def euler_versus_time(self, quaternion_ref_array, quaternion_array):

    #     """
    #     This function consists of two distinct parts: one shows the actual Euler angles and 
    #     the commanded both plotted in the same graph. The next shows the offset of the Euler
    #     angles over time, compared to the maximum allowed offset of 0.1degrees during the
    #     Science & Navigation phase.
    #     """

    #     rot = Rotation()

    #     euler_ref_vis = np.empty((3, len(self.t)))
    #     euler_vis = np.empty((3, len(self.t)))

    #     for index in range(len(self.t)):

    #         # Extract quaternion value for each time step
    #         qwr, qxr, qyr, qzr = quaternion_ref_array[:, index]
    #         qw, qx, qy, qz = quaternion_array[:, index]

    #         # Append do visualisation array
    #         euler_ref_vis[:, index] = np.rad2deg(rot.quaternion_to_euler(qwr, qxr, qyr, qzr))
    #         euler_vis[:, index] = np.rad2deg(rot.quaternion_to_euler(qw, qx, qy, qz))

    #     plt.figure(1)
    #     plt.figure(figsize=(15, 5))
    #     plt.rcParams.update({"font.size": 16})
    #     plt.plot(
    #         self.t,
    #         euler_ref_vis[0, :],
    #         color="red",
    #         linestyle="dashed",
    #         label="Commanded",
    #     )

    #     plt.plot(self.t, euler_vis[0, :], label="Actual")
    #     plt.xlabel("Time since epoch [s]")
    #     plt.ylabel("$\phi$ [deg]")
    #     plt.title("Roll angle commanded and actual signal versus time")
    #     plt.legend()
    #     plt.show()

    #     plt.figure(2)
    #     plt.figure(figsize=(15, 5))
    #     plt.rcParams.update({"font.size": 16})
    #     plt.plot(
    #         self.t,
    #         euler_ref_vis[1, :],
    #         color="red",
    #         linestyle="dashed",
    #         label="Commanded",
    #     )

    #     plt.plot(self.t, euler_vis[1, :], label="Actual")
    #     plt.xlabel("Time since epoch [s]")
    #     plt.ylabel("$\\theta$ [deg]")
    #     plt.title("Pitch angle commanded and actual signal versus time")
    #     plt.legend()
    #     plt.show()

    #     plt.figure(3)
    #     plt.figure(figsize=(15, 5))
    #     plt.rcParams.update({"font.size": 16})
    #     plt.plot(
    #         self.t,
    #         euler_ref_vis[2, :],
    #         color="red",
    #         linestyle="dashed",
    #         label="Commanded",
    #     )

    #     plt.plot(self.t, euler_vis[2, :], label="Actual")
    #     plt.xlabel("Time since epoch [s]")
    #     plt.ylabel("$\psi$ [deg]")
    #     plt.title("Yaw angle commanded and actual signal versus time")
    #     plt.legend()
    #     plt.show()

    #     # Offset

    #     euler_offset_vis = np.abs(euler_vis - euler_ref_vis)
    #     acc_req  = np.full(len(self.t), 0.1)

    #     plt.figure(4)
    #     plt.figure(figsize=(15,5))
    #     plt.rcParams.update({"font.size": 16})
    #     plt.plot(
    #         self.t,
    #         acc_req,
    #         color="red",
    #         linestyle="dashed",
    #         label="ADCS requirement",
    #     )
    #     plt.plot(self.t, euler_offset_vis[0, :], label="Actual offset")
    #     plt.xlabel("Time since epoch [s]")
    #     plt.ylabel("$\Delta$$\phi$ [deg]")
    #     plt.title("Roll angle offset versus time")
    #     plt.legend()
    #     plt.show()

    #     plt.figure(5)
    #     plt.figure(figsize=(15,5))
    #     plt.rcParams.update({"font.size": 16})
    #     plt.plot(
    #         self.t,
    #         acc_req,
    #         color="red",
    #         linestyle="dashed",
    #         label="ADCS requirement",
    #     )
    #     plt.plot(self.t, euler_offset_vis[1, :], label="Actual offset")
    #     plt.xlabel("Time since epoch [s]")
    #     plt.ylabel("$\Delta$$\\theta$ [deg]")
    #     plt.title("Pitch angle offset versus time")
    #     plt.legend()
    #     plt.show()

    #     plt.figure(6)
    #     plt.figure(figsize=(15,5))
    #     plt.rcParams.update({"font.size": 16})
    #     plt.plot(
    #         self.t,
    #         acc_req,
    #         color="red",
    #         linestyle="dashed",
    #         label="ADCS requirement",
    #     )
    #     plt.plot(self.t, euler_offset_vis[2, :], label="Actual offset")
    #     plt.xlabel("Time since epoch [s]")
    #     plt.ylabel("$\Delta$$\psi$ [deg]")
    #     plt.title("Yaw angle offset versus time")
    #     plt.legend()
    #     plt.show()

    #     plt.figure(8)
    #     plt.figure(figsize=(15, 5))
    #     plt.rcParams.update({"font.size": 16})
    #     plt.plot(self.t, acc_req, color="red", linestyle="dashed", label="ADCS requirement")
    #     plt.plot(self.t, euler_offset_vis[0, :], label="Roll offset ($\Delta\phi$)")
    #     plt.plot(self.t, euler_offset_vis[1, :], label="Pitch offset ($\Delta\\theta$)")
    #     plt.plot(self.t, euler_offset_vis[2, :], label="Yaw offset ($\Delta\psi$)")
    #     plt.xlabel("Time since epoch [s]")
    #     plt.ylabel("Angle Offset [deg]")
    #     plt.title("Angle Offset versus Time for Roll, Pitch, and Yaw")
    #     plt.legend()
    #     plt.show()

    #     # Accuracy
    #     plt.figure(7)
    #     plt.figure(figsize=(15,5))
    #     plt.rcParams.update({"font.size": 16})
    #     plt.plot(
    #         self.t,
    #         np.full(len(self.t), 1),
    #         color="red",
    #         linestyle="dashed",
    #         label="ADCS requirement",
    #     )
    #     plt.plot(self.t, acc_req / euler_offset_vis[0, :], label="$\phi_{max}$ / $\phi_{err}$")
    #     plt.plot(self.t, acc_req / euler_offset_vis[1, :], label="$\\theta_{max}$ / $\\theta_{err}$")
    #     plt.plot(self.t, acc_req / euler_offset_vis[2, :], label="$\\psi_{max}$ / $\\psi_{err}$")
    #     plt.ylim(0,10)
    #     plt.xlabel("Time since epoch [s]")
    #     plt.ylabel("Accuracy [-]")
    #     plt.title("Accuracy of all Euler angles over time")
    #     plt.legend()
    #     plt.show()

    def quaternion_versus_time(self, quaternion_ref_array, quaternion_array):

        # Setup for quaternion error array

        pd = PID(Constants.inertia_matrix)
        # Setup for quaternion error array
        qe = np.zeros_like(quaternion_array)  # Initialize error quaternion array with the same shape as quaternion_array

        # Iterate over each time step to compute the error quaternion
        for i in range(quaternion_array.shape[1]):  # Loop over the time array's length
            qe[:, i] = pd.quaternion_error(quaternion_array[:, i], quaternion_ref_array[:, i])  # Compute quaternion error for each time step
            qe[0,i] = quaternion_array[0,i] - quaternion_ref_array[0,i]

        # Plot for quaternion q1
        plt.figure(0)
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(
            self.t,
            quaternion_array[0, :],
            linewidth=1.5,  # Same line width as Commanded
            label="Actual",
        )
        plt.plot(
            self.t,
            quaternion_ref_array[0, :],
            color="red",
            linestyle=(0, (5, 10)),  # Large dashes
            linewidth=1.5,  # Set line width to match Actual line
            label="Commanded",
        )
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$q_w$ [-]", fontsize=16)
        plt.xlim(left=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16)  # Legend font size
        plt.show()

        # Plot for quaternion q1
        plt.figure(1)
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(
            self.t,
            quaternion_array[1, :],
            linewidth=1.5,  # Same line width as Commanded
            label="Actual",
        )
        plt.plot(
            self.t,
            quaternion_ref_array[1, :],
            color="red",
            linestyle=(0, (5, 10)),  # Large dashes
            linewidth=1.5,  # Set line width to match Actual line
            label="Commanded",
        )
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$q_1$ [-]", fontsize=16)
        plt.xlim(left=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16)  # Legend font size
        plt.show()

        # Plot for quaternion q2
        plt.figure(2)
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(
            self.t,
            quaternion_array[2, :],
            linewidth=1.5,  # Same line width as Commanded
            label="Actual",
        )
        plt.plot(
            self.t,
            quaternion_ref_array[2, :],
            color="red",
            linestyle=(0, (5, 10)),  # Large dashes
            linewidth=1.5,  # Set line width to match Actual line
            label="Commanded",
        )
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$q_2$ [-]", fontsize=16)
        plt.xlim(left=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16)  # Legend font size
        plt.show()


        # Plot for quaternion q3
        plt.figure(2)
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(
            self.t,
            quaternion_array[3, :],
            linewidth=1.5,  # Same line width as Commanded
            label="Actual",
        )
        plt.plot(
            self.t,
            quaternion_ref_array[3, :],
            color="red",
            linestyle=(0, (5, 10)),  # Large dashes
            linewidth=1.5,  # Set line width to match Actual line
            label="Commanded",
        )
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$q_3$ [-]", fontsize=16)
        plt.xlim(left=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16)  # Legend font size
        plt.show()

        # Plot for quaternion error components
        plt.figure(figsize=(15, 5))
        # Plot all components of qe
        plt.plot(self.t, qe[0, :], label="$q_{e,w}$", linestyle="solid", color="blue")
        plt.plot(self.t, qe[1, :], label="$q_{e,1}$", linestyle="dashed", color="red")
        plt.plot(self.t, qe[2, :], label="$q_{e,2}$", linestyle="dotted", color="green")
        plt.plot(self.t, qe[3, :], label="$q_{e,3}$", linestyle="dashdot", color="orange")
        # Axis labels
        plt.xlabel("Time [s]", fontsize=16)
        plt.ylabel("$q_{e}$ [-]", fontsize=16)
        # Set x-axis limits
        plt.xlim(left=0)
        # Customize ticks
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
        # Add legend
        plt.legend(fontsize=16)
        # Show the plot
        plt.show()

        # Plot for quaternion relative error components (in percentages)
        plt.figure(figsize=(15, 5))

        # Compute relative error in percentages
        relative_error_0 = np.abs(qe[0, :] / quaternion_array[0, :]) * 100
        relative_error_1 = np.abs(qe[1, :] / quaternion_array[1, :]) * 100
        relative_error_2 = np.abs(qe[2, :] / quaternion_array[2, :]) * 100
        relative_error_3 = np.abs(qe[3, :] / quaternion_array[3, :]) * 100


        # Plot all relative error components
        plt.plot(self.t, relative_error_0, label="$q_{e,w} / q_{w}$ (%)", linestyle="solid", color="blue")
        plt.plot(self.t, relative_error_1, label="$q_{e,1} / q_{1}$ (%)", linestyle="dashed", color="red")
        plt.plot(self.t, relative_error_2, label="$q_{e,2} / q_{2}$ (%)", linestyle="dotted", color="green")
        plt.plot(self.t, relative_error_3, label="$q_{e,3} / q_{3}$ (%)", linestyle="dashdot", color="orange")

        # Axis labels
        plt.xlabel("Time [s]", fontsize=16)
        plt.ylabel("Relative Error [%]", fontsize=16)
        plt.yscale('symlog', linthresh=1e-8)
        # Set x-axis limits
        # plt.xlim(left=0)

        # Customize ticks
        plt.tick_params(axis='both', which='major', labelsize=16)

        # Add legend
        plt.legend(fontsize=16)

        # Show the plot
        plt.show()



    def disturbance_torque_versus_time(self, T_d, T_GG, T_SRP):

        # Apply seaborn style
        plt.style.use('seaborn-v0_8')

        # Disturbance torques
        plt.figure(figsize=(15, 5))
        plt.plot(self.t / (24*3600), T_d[0, :], label="$T_{d,x}$", linestyle="dashed", color="red")
        plt.plot(self.t / (24*3600), T_d[1, :], label="$T_{d,y}$", linestyle="dotted", color="blue")
        plt.plot(self.t / (24*3600), T_d[2, :], label="$T_{d,z}$", linestyle="solid", color="green")
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$T_d$ [Nm]", fontsize=16)
        plt.xlim(left = 0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
        plt.legend(fontsize=16)
        plt.show()

        # Gravity gradient torques
        plt.figure(figsize=(15, 5))
        plt.plot(self.t / (24*3600), T_GG[0, :], label="$T_{GG,x}$", linestyle="dashed", color="red")
        plt.plot(self.t / (24*3600), T_GG[1, :], label="$T_{GG,y}$", linestyle="dotted", color="blue")
        plt.plot(self.t / (24*3600), T_GG[2, :], label="$T_{GG,z}$", linestyle="solid", color="green")
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$T_{GG}$ [Nm]", fontsize=16)
        plt.xlim(left = 0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
        plt.legend(fontsize=16)
        plt.show()

        # Solar radiation pressure torques
        plt.figure(figsize=(15, 5))
        plt.plot(self.t / (24*3600), T_SRP[0, :], label="$T_{SRP,x}$", linestyle="dashed", color="red")
        plt.plot(self.t / (24*3600), T_SRP[1, :], label="$T_{SRP,y}$", linestyle="dotted", color="blue")
        plt.plot(self.t / (24*3600), T_SRP[2, :], label="$T_{SRP,z}$", linestyle="solid", color="green")
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$T_{SRP}$ [Nm]", fontsize=16)
        plt.xlim(left = 0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
        plt.legend(fontsize=16)
        plt.show()

    def control_torque_versus_time(self, T_c):

        # Apply seaborn style
        plt.style.use('seaborn-v0_8')

        # Create a single plot with all components of control torque
        plt.figure(figsize=(15, 5))
        plt.plot(self.t, T_c[0, :], label="$T_{c,x}$", linestyle="dashed", color="red")
        plt.plot(self.t, T_c[1, :], label="$T_{c,y}$", linestyle="dotted", color="blue")
        plt.plot(self.t, T_c[2, :], label="$T_{c,z}$", linestyle="solid", color="green")
        plt.xlabel("Time [s]", fontsize=16)
        plt.ylabel("$T_c$ [Nm]", fontsize=16)
        plt.xlim(left = 0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
        plt.legend(fontsize=16)  # Add labels for x, y, z components
        plt.show()

    def RW_torque_per_wheel(self, torque_matrix):
        """
        Visualizes the torque values for each reaction wheel and their total torque.

        Args:
            torque_matrix (numpy array): 4xlen(self.t) array with torque values for the reaction wheels.

        """
        # Apply seaborn style
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({"font.size": 16})

        # Plot torque for individual reaction wheels and total torque
        plt.figure(figsize=(15, 5))
        for i in range(4):  # Loop through the 4 reaction wheels
            plt.plot(self.t / (24 * 3600), torque_matrix[i, :], label=f"RW {i + 1}")

        # Labels and settings
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$T_{rw}$ [Nm]", fontsize=16)
        # plt.xlim(left=0)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16, loc='upper left')
        plt.show()


    def RW_PE_versus_time(self, PE, select):
        # Apply seaborn style
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({"font.size": 16})

        # Define labels based on selection
        if select == "P":
            y_label = "$P_{rw}$ [W]"
        elif select == "E":
            y_label = "$E_{rw}$ [J]"
        else:
            raise ValueError("Invalid 'select' argument. Use 'P' for power or 'E' for energy.")

        # Plot power/energy for individual reaction wheels and total
        plt.figure(figsize=(15, 5))
        for i in range(4):  # Loop through the 4 reaction wheels
            plt.plot(self.t / (24 * 3600), PE[i, :], label=f"RW {i + 1}")
        
        # Plot total power/energy
        total_PE = np.sum(PE, axis=0)
        plt.plot(self.t / (24 * 3600), total_PE, label="Total", linestyle="dashed", color="black", linewidth=2)

        # Labels and settings
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        # plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16, loc='upper left')
        plt.show()

    def thruster_PE_vs_time(self, PE, select):
        num_thrusters = PE.shape[0]  # Determine the number of thrusters from the array shape

        # Apply seaborn style
        plt.style.use('seaborn-v0_8')

        for i in range(num_thrusters):
            plt.figure(figsize=(5, 5))  # Individual plot size
            plt.plot(self.t / (24*3600), PE[i, :])
            plt.xlabel("Time [days]", fontsize=14)
            plt.xlim(left = 0)
            if select == "P":
                plt.ylabel(f"$P_{{thrust,{i+1}}}$ [W]", fontsize=14)
            elif select == "E":
                plt.ylabel(f"$E_{{thrust,{i+1}}}$ [J]", fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.show()

        # Combined plot for total power or energy of all thrusters
        plt.figure(figsize=(15, 5))  # Combined plot size
        total_PE = np.sum(PE, axis=0)  # Summing up all thrusters' power or energy over time
        for i in range(num_thrusters):
            plt.plot(self.t/ (24*3600), PE[i, :], label=f'Thruster {i+1}')
        plt.plot(self.t/ (24*3600), total_PE, label='Total', linestyle='--', linewidth=2.0, color = 'black')
        plt.xlabel("Time [days]", fontsize=16)
        plt.xlim(left = 0)
        if select == "P":
            plt.ylabel("$P_{thrust}$ [W]", fontsize=16)
        elif select == "E":
            plt.ylabel("$E_{thrust}$ [J]", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16)
        plt.show()

    # TO DO
    def combined_PE_vs_time(self, RW_PE, Thruster_PE, select):
        plt.rcParams.update({"font.size": 16})
        
        # Calculate the total power or energy over time for reaction wheels and thrusters
        total_RW_PE = np.sum(RW_PE, axis=0)
        total_Thruster_PE = np.sum(Thruster_PE, axis=0)
        
        plt.figure(figsize=(15, 5))
        
        # Plotting the total from reaction wheels
        plt.plot(self.t / (24*3600), total_RW_PE, label='Total Reaction Wheels')
        
        # Plotting the total from thrusters
        plt.plot(self.t / (24*3600), total_Thruster_PE, label='Total Thrusters')
        plt.yscale('log')
        plt.xlabel("Time [days]")
        if select == "P":
            plt.ylabel("Total Power [W]")
            plt.title("Total power usage of reaction wheels and thrusters over time")
        elif select == "E":
            plt.ylabel("Total Energy [J]")
            plt.title("Total energy consumption of reaction wheels and thrusters over time")
        
        plt.legend()
        plt.show()

    def omega_versus_time(self, omega_array):

        # Apply seaborn style
        plt.style.use('seaborn-v0_8')

        # Create a single plot with all angular velocity components
        plt.figure(figsize=(15, 5))
        plt.plot(self.t, omega_array[0, :], label="$\\omega_x$", linestyle="dashed", color="red")
        plt.plot(self.t, omega_array[1, :], label="$\\omega_y$", linestyle="dotted", color="blue")
        plt.plot(self.t, omega_array[2, :], label="$\\omega_z$", linestyle="solid", color="green")
        plt.xlabel("Time [s]", fontsize=16)
        plt.ylabel("$\\omega$ [rad/s]", fontsize=16)
        plt.xlim(left = 0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
        plt.legend(fontsize=16)  # Add labels for x, y, z components
        plt.show()

    def h_versus_time(self, h_array):
        # Import Seaborn style
        plt.style.use('seaborn-v0_8')

        # Reaction wheel saturation limits
        h_sat_1_2_3 = np.full(len(self.t), 0.1)  # Saturation limit for wheels 1, 2, and 3 [Nms]
        h_sat_4 = np.full(len(self.t), 0.05)  # Saturation limit for wheel 4 [Nms]

        # Combine angular momentum for reaction wheels 1, 2, and 3
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(self.t / (24 * 3600), h_array[0, :], label="RW 1")
        plt.plot(self.t / (24 * 3600), h_array[1, :], label="RW 2")
        plt.plot(self.t / (24 * 3600), h_array[2, :], label="RW 3")
        plt.plot(self.t / (24 * 3600), h_sat_1_2_3, color="red", linestyle="dashed", label="Saturation limit (+/- 0.1 Nms)")
        plt.plot(self.t / (24 * 3600), -h_sat_1_2_3, color="red", linestyle="dashed")
        plt.yscale('symlog', linthresh=1e-8)
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$h_{rw}$ [Nms]", fontsize=16)
        plt.xlim(left=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16, loc = 'upper left')
        plt.show()

        # Angular momentum for reaction wheel 4
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(self.t / (24 * 3600), h_array[3, :], label="RW 4")
        plt.plot(self.t / (24 * 3600), h_sat_4, color="red", linestyle="dashed", label="Saturation limit (+/- 0.05 Nms)")
        plt.plot(self.t / (24 * 3600), -h_sat_4, color="red", linestyle="dashed")
        plt.yscale('symlog', linthresh=1e-8)
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$h_{rw}$ [Nms]", fontsize=16)
        plt.xlim(left=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(fontsize=16)
        plt.show()
        

    def thrust_values(self, thruster_force_values):
        # Apply seaborn style
        plt.style.use('seaborn-v0_8')

        # Number of thrusters per graph
        thrusters_per_graph = 6
        num_thrusters = thruster_force_values.shape[0]
        num_graphs = (num_thrusters + thrusters_per_graph - 1) // thrusters_per_graph  # Calculate number of graphs

        # Loop through groups of thrusters
        for group_idx in range(num_graphs):
            start_idx = group_idx * thrusters_per_graph
            end_idx = min((group_idx + 1) * thrusters_per_graph, num_thrusters)

            # Plot combined graph for this group of thrusters
            plt.figure(figsize=(15, 5))
            for i in range(start_idx, end_idx):
                plt.plot(
                    self.t/ (24*3600),
                    thruster_force_values[i, :],
                    label=f"Thruster {i + 1}"
                )
            plt.xlabel("Time [days]", fontsize=16)
            plt.ylabel("$F_{thrust}$ [N]", fontsize=16)
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.tick_params(axis='both', which='major', labelsize=16)
            ax = plt.gca()  # Get the current axis
            ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
            plt.legend(fontsize=16, loc='upper left', ncol=2)
            plt.show()

    def thrust_values_difference(self, thruster_force_values):
        # Apply seaborn style
        plt.style.use('seaborn-v0_8')

        # Number of thrusters per graph
        thrusters_per_graph = 6
        num_thrusters = thruster_force_values.shape[0]
        num_graphs = (num_thrusters + thrusters_per_graph - 1) // thrusters_per_graph  # Calculate number of graphs

        # Loop through groups of thrusters
        for group_idx in range(num_graphs):
            start_idx = group_idx * thrusters_per_graph
            end_idx = min((group_idx + 1) * thrusters_per_graph, num_thrusters)

            # Plot combined graph for this group of thrusters
            plt.figure(figsize=(15, 5))
            for i in range(start_idx, end_idx):
                plt.plot(
                    self.t/ (24*3600),
                    thruster_force_values[i, :],
                    label=f"Thruster {i + 1}"
                )
            plt.xlabel("Time [days]", fontsize=16)
            plt.ylabel("$\Delta F_{thrust}$ [N]", fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=16)
            ax = plt.gca()  # Get the current axis
            ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the 1e-6
            plt.legend(fontsize=16, loc='upper left', ncol=2)
            plt.show()


    def impulse_versus_time(self, impulse):
        # Apply seaborn style
        plt.style.use('seaborn-v0_8')

        # Define custom behavior for 8-thruster setup
        num_thrusters = impulse.shape[0]
        line_styles = ['solid'] * 6 + ['dashed'] * (num_thrusters - 6)  # First 6 solid, rest dashed

        # Plot
        plt.figure(figsize=(15, 5))
        for i in range(num_thrusters):
            plt.plot(
                self.t / (24 * 3600), 
                impulse[i, :], 
                label=f"Thruster {i + 1}", 
                linestyle=line_styles[i % len(line_styles)]  # Use solid for first 6, dashed for 7 and 8
            )
        
        # Labels and settings
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$J_{thrust}$ [Ns]", fontsize=16)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tick_params(axis='both', which='major', labelsize=16)

        # Adjust y-axis offset text font size
        ax = plt.gca()
        ax.yaxis.get_offset_text().set_fontsize(16)

        # Legend with multiple columns
        plt.legend(fontsize=16, loc='upper left', ncol=2)
        plt.show()

    def impulse_versus_time_2(self, impulse):
        # Apply seaborn style
        plt.style.use('seaborn-v0_8')

        # Total number of thrusters
        num_thrusters = impulse.shape[0]

        # Split into two groups
        thrusters_per_plot = 6
        num_plots = int(np.ceil(num_thrusters / thrusters_per_plot))

        for plot_idx in range(num_plots):
            start_idx = plot_idx * thrusters_per_plot
            end_idx = min((plot_idx + 1) * thrusters_per_plot, num_thrusters)

            # Plot each group of thrusters
            plt.figure(figsize=(15, 5))
            for i in range(start_idx, end_idx):
                plt.plot(
                    self.t / (24 * 3600),
                    impulse[i, :],
                    label=f"Thruster {i + 1}"  # Thruster numbering starts from 1
                )
            
            # Labels and settings
            plt.xlabel("Time [days]", fontsize=16)
            plt.ylabel("$J_{thrust}$ [Ns]", fontsize=16)
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.tick_params(axis='both', which='major', labelsize=16)

            # Adjust y-axis offset text font size
            ax = plt.gca()
            ax.yaxis.get_offset_text().set_fontsize(16)

            # Legend with multiple columns if needed
            plt.legend(fontsize=16, loc='upper left', ncol=2)
            plt.show()

    def half_cone_versus_time(self, cone_3, cone_4, cone_base_case):

        # Apply seaborn style
        plt.style.use('seaborn-v0_8')
        requirement = np.ones(len(cone_3)) * 0.18

        # Plot combined graph for all configurations
        plt.figure(figsize=(15, 5))  # Set combined graph size
        plt.plot(self.t / (24 * 3600), cone_3, label="STF Conf. 3", linestyle="-")
        plt.plot(self.t / (24 * 3600), cone_4, label="STF Conf. 4", linestyle="--")
        plt.plot(self.t / (24 * 3600), cone_base_case, label="Base case", linestyle="-.")
        plt.plot(self.t / (24 * 3600), requirement, label="ADCS-01 requirement", color="red", linestyle="dashed")

        # Add labels and customize ticks
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$\\beta$ [$\\degree$]", fontsize=16)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the offset text

        # Add legend
        plt.legend(fontsize=16)

        # Show plot
        plt.show()

        # Plot combined graph for all configurations
        plt.figure(figsize=(15, 5))  # Set combined graph size
        plt.plot(self.t / (24 * 3600), cone_3, label="STF Conf. 3", linestyle="-")
        plt.plot(self.t / (24 * 3600), cone_4, label="STF Conf. 4", linestyle="--")
        plt.plot(self.t / (24 * 3600), cone_base_case, label="Base case", linestyle="-.")

        # Add labels and customize ticks
        plt.xlabel("Time [days]", fontsize=16)
        plt.ylabel("$\\beta$ [$\\degree$]", fontsize=16)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax = plt.gca()  # Get the current axis
        ax.yaxis.get_offset_text().set_fontsize(16)  # Adjust the font size of the offset text

        # Add legend
        plt.legend(fontsize=16)

        # Show plot
        plt.show()


def simulation(
    data_file_CAPSTONE,
    data_file_Sun,
    omega_0,
    kp,
    kd,
    ks,
    T_matrix,
    matrixnumber
):
    # Load data files
    data_CAPSTONE = np.loadtxt(data_file_CAPSTONE)
    data_Sun = np.loadtxt(data_file_Sun)

    # Load data files
    data_CAPSTONE = np.loadtxt(data_file_CAPSTONE)
    data_Sun = np.loadtxt(data_file_Sun)

    # Extract metadata from data_file_CAPSTONE filename using regular expressions
    # Looks for date and time in the format YYYY-MM-DD HH:MM
    filename = os.path.basename(data_file_CAPSTONE)
    datetime_matches = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', filename)

    if len(datetime_matches) >= 2:
        startdate = datetime_matches[0].replace(":", "-").replace(" ", "_")
        enddate = datetime_matches[1].replace(":", "-").replace(" ", "_")
    else:
        raise ValueError("Could not extract start and end date-times from the filename.")

    # Define save directory
    save_dir = "/Users/Pieter/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Research Phase/lunar_CubeSat/results/data"
    os.makedirs(save_dir, exist_ok=True)

    # Create time array based on the input data files and check if they cover the same epochs
    time_array = data_CAPSTONE[:, 0]
    time_array_check = data_Sun[:, 0]
    if time_array.all() != time_array_check.all():
        raise ValueError(
            "Epochs of the data files are not the same; check ephemeris retrieval."
        )

    # Obtain position data over time
    position_CAPSTONE = data_CAPSTONE[:, 1:4]
    position_Sun = data_Sun[:, 1:4]

    # Retrieve time step from data file
    dt = time_array[1] - time_array[0]

    vis = Visualization(time_array)
    const = Constants()
    rot = Rotation()
    pd = PID(const.inertia_matrix)
    dist = DisturbanceTorques(const.inertia_matrix)

    # Initialize individual vectors for iteration
    # q = quaternion_0_vector # For custom initial condition
    q = pd.reference_quaternion_paper(position_CAPSTONE[0,:], position_Sun[0,:]) # For standard initial position
    omega = omega_0

    # Dummy values for PID
    factor_p = kp
    factor_i = 0
    factor_d = kd
    factor_s = ks

    # Create arrays
    k_p = np.array([1, 1, 1]) * factor_p
    k_i = np.array([1, 1, 1]) * factor_i
    k_d = np.array([1, 1, 1]) * factor_d
    k_s = np.array([1, 1, 1]) * factor_s


    # Initialize storage arrays for visualization
    q_ref_vis = np.empty((4, len(time_array)))
    q_vis = np.empty((4, len(time_array)))
    T_GG_vis = np.empty((3, len(time_array)))
    T_SRP_vis = np.empty((3, len(time_array)))
    T_d_vis = np.empty((3, len(time_array)))
    T_c_vis = np.empty((3, len(time_array)))
    T_RW_vis = np.empty((3, len(time_array)))
    P_RW_vis = np.empty((3, len(time_array)))
    E_RW_vis = np.empty((3, len(time_array)))
    omega_vis = np.empty((3, len(time_array)))
    h_vis = np.empty((3, len(time_array)))
    thrust_vis = np.empty((len(T_matrix[0]), len(time_array)))
    impulse_vis = np.empty((len(T_matrix[0]), len(time_array)))
    P_thrust_vis = np.empty((len(T_matrix[0]), len(time_array)))
    E_thrust_vis = np.empty((len(T_matrix[0]), len(time_array)))
    half_cone_offset_vis = np.empty((len(time_array)))

    # Index count and previous q_ref value
    index = 0
    previous_q_ref = None

    # Initiate angular momentum, reaction wheel energy, thrust energy, impulse
    h = 0
    E_RW = 0
    E_thrust = 0
    impulse = np.zeros(len(T_matrix[0]))

    for t in time_array:

        # LUMIO paper-based reference quaternion update
        q_ref = pd.reference_quaternion_paper(position_CAPSTONE[index,:], position_Sun[index,:])

        # Check quaternion continuity; do we need to flip signs?
        if previous_q_ref is not None:
            if np.dot(q_ref, previous_q_ref) < 0:
                q_ref = -q_ref
        
        # Store q_ref for next iteration
        previous_q_ref = q_ref

        # Half-cone requirement validation
        y_panel_body = np.array([0,-1,0])
        y_panel_inert = np.linalg.inv(rot.quaternion_321_rotation(q[0], q[1], q[2], q[3])) @ y_panel_body
        y_panel_inert /= np.linalg.norm(y_panel_inert)
        Moon_pointing_vector = - position_CAPSTONE[index,:]
        Moon_pointing_vector /= np.linalg.norm(Moon_pointing_vector)
        half_cone_offset = np.rad2deg(np.arccos(np.clip(np.dot(y_panel_inert, Moon_pointing_vector), -1.0, 1.0)))

        # Calculate real-time disturbance torques
        T_GG = dist.GGMoon(q, position_CAPSTONE[index, :])
        T_SRP = dist.SRP_deployed(q, position_Sun[index, :], position_CAPSTONE[index, :])
        # Dummy input zeros
        T_d = T_GG + T_SRP

        T_c = pd.control_torque(q, q_ref, omega, k_p, k_i, k_d, k_s, dt)
        
        ######### RW ANALYSIS #######################
        # Calculate incremental increase in angular momentum of the momentum wheel
        h = const.inertia_matrix_undeployed @ omega
        # Assembly configuration matrix, each reaction wheel covers one axis
        A_RW = np.array([[1,0,0], [0,1,0], [0,0,1]])
        # Torque imposed on RW assembly, nominal mode
        T_ass = np.linalg.pinv(A_RW) @ (-T_c + np.cross(A_RW @ h, omega))
        # Torque generated by the RW set, total
        T_RW = - (A_RW @ T_ass + np.cross(omega, A_RW @ h))
        # Power consumption at this time step, linear relation assumed
        P_RW = const.P_peak_max / const.T_RW_max * np.abs(T_RW)
        # Total energy consumption, power integrated over time
        E_RW += P_RW * dt
        #############################################

        ######### THRUSTER ANALYSIS #################
        # Linear Programming Solution
        thrust = cp.Variable(len(T_matrix[0]))
        constraints = [T_matrix @ thrust == T_c, thrust >= 0, thrust <= const.F_SSP_max]
        objective = cp.Minimize(cp.sum(thrust))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # # Check the solution status
        # if problem.status == cp.OPTIMAL:
        #     print("Solution found!")
        #     thrust_value = thrust.value
        #     print("Thrust values:", thrust_value)
        # elif problem.status == cp.INFEASIBLE:
        #     print("No solution exists (problem is infeasible).")
        # elif problem.status == cp.UNBOUNDED:
        #     print("Problem is unbounded (no finite solution).")
        # else:
        #     print(f"Solver status: {problem.status}")

        if thrust.value is None:
            thrust_value = np.zeros(len(T_matrix[0]))
        else:
            thrust_value = np.clip(thrust.value, 0, None)  # Set all values < 0 to 0

        # Add uncertainty
        sigma = 0.05  # 5% uncertainty introduced, random value
        thrust_value_actual = thrust_value.copy()  # Start with the original values
        non_zero_indices = thrust_value > 0  # Find indices where thrust value is positive

        # Apply noise only to non-zero values
        thrust_value_actual[non_zero_indices] = np.clip(
            np.random.normal(
                loc=thrust_value[non_zero_indices],
                scale=sigma * thrust_value[non_zero_indices]
            ),
            (1 - sigma) * thrust_value[non_zero_indices],
            (1 + sigma) * thrust_value[non_zero_indices]
        )

        # Update impulse value
        impulse += thrust_value_actual * dt

        # Power analysis
        # From data sheet: Input Power linear relation with thrust output, from T = 0 to T = 200 [muN] and P = 0 to 20 [W]
        # Let power input be as in original thrust case, since uncertainty comes from the output only
        P_thrust = const.P_SSP_max / const.F_SSP_max * thrust_value
        E_thrust += P_thrust * dt
        #############################################

        # Append to visualisation arrays
        q_ref_vis[:, index] = q_ref
        q_vis[:, index] = q
        T_d_vis[:, index] = T_d
        T_GG_vis[:, index] = T_GG
        T_SRP_vis[:, index] = T_SRP
        T_c_vis[:, index] = T_c
        T_RW_vis[:, index] = T_RW
        P_RW_vis[:, index] = P_RW
        E_RW_vis[:, index] = E_RW
        omega_vis[:, index] = omega
        h_vis[:, index] = h
        half_cone_offset_vis[index] = half_cone_offset
        thrust_vis[:,index] = thrust_value_actual
        impulse_vis[:,index] = impulse
        P_thrust_vis[:,index] = P_thrust
        E_thrust_vis[:,index] = E_thrust

        # Update to t = 1
        # Integrate omega and quaternion
        omega_new = pd.rk4_integrator(pd.derivative_omega, omega, dt, T_d, T_c)
        q_new = pd.rk4_integrator(pd.derivative_quaternion, q, dt, omega)
        q = q_new / np.linalg.norm(q_new)
        omega = omega_new

        # Update index
        index += 1
        print(index)
    
    # At the end of the simulation loop, after populating the visualization arrays
    total_PE = np.sum(P_thrust_vis, axis=0)  # Summing up all thrusters' power over time
    total_E = np.sum(E_thrust_vis[:, -1])  # Total energy is the last value of summation

    # Save each "vis" array
    vis_arrays = {
        "q_ref_vis": q_ref_vis,
        "q_vis": q_vis,
        "T_GG_vis": T_GG_vis,
        "T_SRP_vis": T_SRP_vis,
        "T_d_vis": T_d_vis,
        "T_c_vis": T_c_vis,
        "T_RW_vis": T_RW_vis,
        "P_RW_vis": P_RW_vis,
        "E_RW_vis": E_RW_vis,
        "omega_vis": omega_vis,
        "h_vis": h_vis,
        "thrust_vis": thrust_vis,
        "impulse": impulse_vis,
        "P_thrust_vis": P_thrust_vis,
        "E_thrust_vis": E_thrust_vis,
        "half_cone_offset_vis": half_cone_offset_vis,
        "time_array": time_array
    }

    for var_name, data in vis_arrays.items():
        file_name = f"{startdate}_{enddate}_{var_name}_{matrixnumber}.dat"
        file_path = os.path.join(save_dir, file_name)
        np.savetxt(file_path, data.T, delimiter=" ", fmt="%.20f")  # Transpose data for column format
    
    # Plot generation
    vis.quaternion_versus_time(q_ref_vis, q_vis)
    # vis.euler_versus_time(q_ref_vis, q_vis)
    # vis.control_torque_versus_time(T_c_vis)
    # vis.RW_PE_versus_time(P_RW_vis, "P")
    # vis.RW_PE_versus_time(E_RW_vis, "E")
    # vis.h_versus_time(h_vis)
    # vis.disturbance_torque_versus_time(T_d_vis, T_GG_vis, T_SRP_vis)
    # vis.omega_versus_time(omega_vis)
    # vis.plot_trajectories(data_file_CAPSTONE)
    vis.thruster_PE_vs_time(P_thrust_vis, "P")
    vis.thruster_PE_vs_time(E_thrust_vis, "E")
    # vis.combined_PE_vs_time(P_RW_vis, P_thrust_vis, "P")
    # vis.combined_PE_vs_time(E_RW_vis, E_thrust_vis, "E")
    vis.thrust_values(thrust_vis)
    
    # Return required values
    return {
        "max_total_power": np.max(total_PE),
        "max_total_energy": total_E
    }

# ############## INPUT ##################################

# ephemeris_time_step = 1/60 # hours
# begin_date = "2023-01-01 00:00"
# end_date = "2023-01-01 00:05"
# control_time_step = 0.001 # seconds

# #######################################################

# eph_CAPSTONE = EphemerisData(
#     Constants.id_CAPSTONE,
#     Constants.location_Moon_centre,
#     ephemeris_time_step,
#     begin_date,
#     end_date,
# )
# eph_CAPSTONE.convert_data_interpolated(eph_CAPSTONE.vectors()[0], control_time_step)

# eph_Sun = EphemerisData(
#     Constants.id_Sun,
#     Constants.location_Moon_centre,
#     ephemeris_time_step,
#     begin_date,
#     end_date,
# )
# eph_Sun.convert_data_interpolated(eph_Sun.vectors()[0], control_time_step)

# # Set-up 1
# tau_11 = np.cross(np.array([0.1, 0.15, 0]), np.array([-1,0,0]))
# tau_12 = np.cross(np.array([-0.1, 0.15, 0]), np.array([1,0,0]))
# tau_13 = np.cross(np.array([0, 0.15, 0.1]), np.array([0,0,-1]))
# tau_14 = np.cross(np.array([0, -0.15, 0.1]), np.array([0,0,-1]))
# tau_15 = np.cross(np.array([0.1, 0, 0.1]), np.array([-1,0,0]))
# tau_16 = np.cross(np.array([-0.1, 0, 0.1]), np.array([1,0,0]))

# # Set-up 2
# tau_21 = np.cross(np.array([0.1, 0.15, 0.1]), np.array([-1,0,0]))
# tau_22 = np.cross(np.array([-0.1, 0.15, 0.1]), np.array([1,0,0]))
# tau_23 = np.cross(np.array([0, 0.15, 0.1]), np.array([0,0,-1]))
# tau_24 = np.cross(np.array([0, -0.15, 0.1]), np.array([0,0,-1]))
# tau_25 = np.cross(np.array([0.1, -0.15, 0.1]), np.array([-1,0,0]))
# tau_26 = np.cross(np.array([-0.1, -0.15, 0.1]), np.array([1,0,0]))

# # Set-up 3
# tau_31 = np.cross(np.array([0.1, 0.15, 0]), np.array([-1,0,0]))
# tau_32 = np.cross(np.array([-0.1, 0.15, 0]), np.array([1,0,0]))
# tau_33 = np.cross(np.array([0, 0.15, 0.1]), np.array([0,0,-1]))
# tau_34 = np.cross(np.array([0, -0.15, 0.1]), np.array([0,0,-1]))
# tau_35 = np.cross(np.array([0.1, 0, 0.1]), np.array([-1,0,0]))
# tau_36 = np.cross(np.array([-0.1, 0, 0.1]), np.array([1,0,0]))
# tau_37 = np.cross(np.array([0.1, -0.15, -0.1]), np.array([0,1,0]))
# tau_38 = np.cross(np.array([-0.1, -0.15, -0.1]), np.array([0,1,0]))

# # Set-up 4
# tau_41 = np.cross(np.array([0.1, -0.15, 0.1]), np.array([0,1,0]))
# tau_42 = np.cross(np.array([0.1, -0.15, 0.1]), np.array([-1,0,0]))
# tau_43 = np.cross(np.array([0.1, -0.15, 0.1]), np.array([0,0,-1]))
# tau_44 = np.cross(np.array([-0.1, -0.15, 0.1]), np.array([1,0,0]))
# tau_45 = np.cross(np.array([-0.1, -0.15, 0.1]), np.array([0,1,0]))
# tau_46 = np.cross(np.array([-0.1, -0.15, 0.1]), np.array([0,0,-1]))
# tau_47 = np.cross(np.array([0.1, -0.15, -0.1]), np.array([-1,0,0]))
# tau_48 = np.cross(np.array([0.1, -0.15, -0.1]), np.array([0,1,0]))
# tau_49 = np.cross(np.array([0.1, -0.15, -0.1]), np.array([0,0,1]))
# tau_410 = np.cross(np.array([-0.1, -0.15, -0.1]), np.array([0,0,1]))
# tau_411 = np.cross(np.array([-0.1, -0.15, -0.1]), np.array([0,1,0]))
# tau_412 = np.cross(np.array([-0.1, -0.15, -0.1]), np.array([1,0,0]))

# # Stack the tau vectors into a 3x6 matrix
# T_matrix_1 = np.column_stack((tau_11, tau_12, tau_13, tau_14, tau_15, tau_16))
# T_matrix_2 = np.column_stack([tau_21, tau_22, tau_23, tau_24, tau_25, tau_26])
# T_matrix_3 = np.column_stack([tau_31, tau_32, tau_33, tau_34,tau_35, tau_36, tau_37, tau_38])
# T_matrix_4 = np.column_stack([tau_41, tau_42, tau_43, tau_44,tau_45, tau_46, tau_47, tau_48, tau_49, tau_410, tau_411, tau_412])
