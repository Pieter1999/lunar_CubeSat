from astroquery.jplhorizons import Horizons
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.linalg import null_space
import pandas as pd


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
    R_Sun = 695700e3 # Sun mean radius [m]
    P_solar = 3.842e26  # Power exerted by the Sun, [W]

    # LUMIO spacecraft data
    inertia_matrix = np.array([[100.9, 0, 0], [0, 25.1, 0], [0, 0, 91.6]]) * 10 ** (
        -2
    )  # deployed inertia matrix, [kg m^2]

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

    def SRP(self, position_Sun_Moon, position_SC_Moon):

        rho_s = 0.6  # These values have been taken from the LUMIO ADCS paper for now; to be verified or adjusted
        rho_d = 0.1

        # First, define surface CoM locations wrt satellite CoM [0,0,0] in body frame
        # Numbering according to ADCS LUMIO paper
        # Different reference frame, same as thruster config. Surface 1 is in positive z-axis, surface 4 is positive x-axis, surface 6 is positive y-axis
        # All values in meter

        S_loc_1 = np.array([0, 0, 0.1])
        S_loc_2 = np.array([0, 0, -0.1])
        S_loc_3 = np.array([-0.1, 0, 0])
        S_loc_4 = np.array([0.1, 0, 0])
        S_loc_5 = np.array([0, -0.1, 0])
        S_loc_6 = np.array([0, 0.1, 0])

        c_p = np.column_stack(
            [S_loc_1, S_loc_2, S_loc_3, S_loc_4, S_loc_5, S_loc_6]
        )  # Centre of pressure locations for calculation
        n_s = np.array([[0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1], [-1, 1, 0, 0, 0, 0]])
        r_S_SC = position_Sun_Moon - position_SC_Moon
        S = np.column_stack(
            [
                (S_loc_1 - r_S_SC) / np.linalg.norm(S_loc_1 - r_S_SC),
                (S_loc_2 - r_S_SC) / np.linalg.norm(S_loc_2 - r_S_SC),
                (S_loc_3 - r_S_SC) / np.linalg.norm(S_loc_3 - r_S_SC),
                (S_loc_4 - r_S_SC) / np.linalg.norm(S_loc_4 - r_S_SC),
                (S_loc_5 - r_S_SC) / np.linalg.norm(S_loc_5 - r_S_SC),
                (S_loc_6 - r_S_SC) / np.linalg.norm(S_loc_6 - r_S_SC),
            ]
        )
        A = np.array([6, 6, 6, 6, 4, 4]) * 10 ** (
            -2
        )  # Surface areas of all labeled surfaces, in m2
        F = np.empty((3, len(A)))

        for i in range(len(A)):
            F_SRP = (
                400
                / Constants.c
                * A[i]
                * np.dot(S.T[i], n_s.T[i])
                * (
                    (1 - rho_s) * S.T[i]
                    + (2 * rho_s * np.dot(S.T[i], n_s.T[i]) + 2 / 3 * rho_d) * n_s.T[i]
                )
            )

            print(
                "Solar radiation pressure for surface " + str(i) + " is " + str(F_SRP)
            )

            if np.dot(S.T[i], n_s.T[i]) > 0:
                F[0, i], F[1, i], F[2, i] = np.cross(c_p[:, i], F_SRP)

            else:
                F[:, i] = np.zeros(3)

        print("Solar radiation torque due to all surfaces is " + str(F))
        print("Total solar radiation torque is " + str(F.sum(axis=1)))

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
    start_date: custom start date, default set to beginning of nominal CAPSTONE mission
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

        np.savetxt(
            "/Users/pieter/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Research Phase/lunar_CubeSat/ephemeris_data/cartesian_coordinates_"
            + body
            + "_"
            + center
            + "_"
            + self.t0
            + "_to_"
            + self.t1 + "_"
            + str(self.time_step) 
            + ".dat",
            cartesian,
        )

        return cartesian

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
            + self.t1 + "_"
            + str(self.time_step)
            + ".dat",
            keplerian,
        )

        return keplerian

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
        qw, qx, qy, qz = quaternion_vector
        q_conj = np.array([qw, -qx, -qy, -qz])
        rot = Rotation()

        q_error_vector = rot.quaternion_product(quaternion_ref_vector, q_conj)

        return q_error_vector

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
        self, quaternion_vector, quaternion_ref_vector, omega_vector, k_p, k_i, k_d, dt
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
        qew_dot, qex_dot, qey_dot, qez_dot = self.derivative_quaternion_error(
            quaternion_vector, quaternion_ref_vector, omega_vector
        )  # quaternion error time derivative components
        k_p_1, k_p_2, k_p_3 = k_p  # proportional gain
        k_i_1, k_i_2, k_i_3 = k_i  # integral gain
        k_d_1, k_d_2, k_d_3 = k_d  # derivative gain

        int_x, int_y, int_z = self.integral_term

        T_c_x = (
            k_p_1 * qex + k_i_1 * int_x + k_d_1 * qex_dot
        )  # control torque in the x, y and z directions
        T_c_y = k_p_2 * qey + k_i_2 * int_y + k_d_2 * qey_dot
        T_c_z = k_p_3 * qez + k_i_3 * int_z + k_d_3 * qez_dot

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
        '''
        Basic Euler integrator.
        Args:
            func: The function to calculate derivatives, with signature func(y, *args)
            y: Current state variable (np.array)
            dt: Time step (float)
            *args: Additional arguments required by `func`
        Returns:
            np.array: Updated state after dt
        '''
        dy = func(y, *args)  # Calculate the derivative
        return y + dy * dt  # Update state

class Visualization:
    """
    Visualization class for result from attitude control.
    """
    def __init__(self, time_array):
        self.t = time_array

    def plot_trajectories(self, *data_files):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        reference_frames = {'Sun': Constants.R_Sun, 'Moon': Constants.R_Moon, 'Earth': Constants.R_Earth, 'CAPSTONE': 0.2}  # Radii in meters
        used_centers = set()

        first_file = True
        all_epochs = None
        
        for data_file in data_files:
            # Extracting information from file name
            parts = data_file.split('_')
            body_name = parts[3]
            center_part = parts[4]
            center = center_part.split('-')[0]

            print(center)

            # Check if center has been used consistently
            if used_centers and center not in used_centers:
                raise ValueError(f"Inconsistent reference systems: {used_centers.pop()} vs {center}. All data must be in the same reference frame.")
            used_centers.add(center)
            
            # Reading data
            data = np.loadtxt(data_file)
            epochs, x, y, z, vx, vy, vz = data.T
            x, y, z = x * Constants.AU, y * Constants.AU, z * Constants.AU  # AU to meters
            
            # Plotting trajectories
            ax.plot(x, y, z, label=body_name)
            
            # Check epochs
            if all_epochs is None:
                all_epochs = epochs
            elif not np.array_equal(all_epochs, epochs):
                raise ValueError("Epochs do not match across data files. Ensure all input files cover the same time periods.")

            
            # Centering reference body
            if center in reference_frames:
                # Plotting central body as a sphere
                radius = reference_frames[center]
                u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
                sphere_x = radius * np.cos(u) * np.sin(v)
                sphere_y = radius * np.sin(u) * np.sin(v)
                sphere_z = radius * np.cos(v)
                ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="grey", alpha=0.5)
                ax.text(0, 0, - 5 * radius, center, color='black', fontsize=8, ha='center')
            
        # Setting limits for equal aspect ratio
        max_radius = max(abs(ax.get_xlim()[0]), abs(ax.get_ylim()[0]), abs(ax.get_zlim()[0]))
        ax.set_xlim([-max_radius, max_radius])
        ax.set_ylim([-max_radius, max_radius])
        ax.set_zlim([-max_radius, max_radius])
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectories in Solar System')
        ax.legend()
        
        plt.show()

    def quaternion_versus_time(self, quaternion_ref_array, quaternion_array):

        plt.figure(1)
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(
            self.t,
            quaternion_ref_array[0, :],
            color="red",
            linestyle="dashed",
            label="Commanded",
        )
        plt.plot(self.t, quaternion_array[0, :], label="Actual")
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("$q_1$ [-]")
        plt.title("First quaternion commanded and actual signal versus time")
        plt.legend()
        # plt.savefig("Figures/linear_q_1.png")
        plt.show()

        plt.figure(2)
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(
            self.t,
            quaternion_ref_array[1, :],
            color="red",
            linestyle="dashed",
            label="Commanded",
        )
        plt.plot(self.t, quaternion_array[1, :], label="Actual")
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("$q_2$ [-]")
        plt.title("Second quaternion commanded and actual signal versus time")
        plt.legend()
        # plt.savefig("Figures/linear_q_2.png")
        plt.show()

        plt.figure(3)
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(
            self.t,
            quaternion_ref_array[2, :],
            color="red",
            linestyle="dashed",
            label="Commanded",
        )
        plt.plot(self.t, quaternion_array[2, :], label="Actual")
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("$q_3$ [-]")
        plt.title("Third quaternion commanded and actual signal versus time")
        plt.legend()
        # plt.savefig("Figures/linear_q_3.png")
        plt.show()

        plt.figure(4)
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(
            self.t,
            quaternion_ref_array[3, :],
            color="red",
            linestyle="dashed",
            label="Commanded",
        )
        plt.plot(self.t, quaternion_array[3, :], label="Actual")
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("$q_4$ [-]")
        plt.title("Fourth quaternion commanded and actual signal versus time")
        plt.legend()
        # plt.savefig("Figures/linear_q_4.png")
        plt.show()

    def disturbance_torque_versus_time(self, T_d, T_GG=None, T_SRP=None):

        T_d_mag = np.sqrt(np.sum(T_d**2, axis=0))
        # if T_GG.any() != None:
        #     T_GG_mag = np.sqrt(np.sum(T_GG**2, axis = 0))
        # if T_SRP.any() != None:
        #     T_SRP_mag = np.sqrt(np.sum(T_SRP**2, axis = 0))

        plt.figure(1)
        plt.rcParams.update({"font.size": 16})
        plt.yscale("log")
        plt.plot(self.t, T_d[0, :], label="Total disturbance")
        # if T_GG.any() != None:
        #     plt.plot(self.t, T_GG[0,:], label="Gravity gradient")
        # if T_SRP.any() != None:
        #     plt.plot(self.t, T_SRP[0,:], label = "Solar radiation pressure")
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Disturbance torques about x-axis [Nm]")
        plt.title("Disturbance torque about the x-axis versus time")
        # plt.savefig("Figures/linear_Tc_q")
        plt.show()

        plt.figure(2)
        plt.rcParams.update({"font.size": 16})
        plt.yscale("log")
        plt.plot(self.t, T_d[1, :], label="Total disturbance")
        # if T_GG.any() != None:
        #     plt.plot(self.t, T_GG[1,:], label="Gravity gradient")
        # if T_SRP.any() != None:
        #     plt.plot(self.t, T_SRP[1,:], label = "Solar radiation pressure")
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Disturbance torques about y-axis [Nm]")
        plt.title("Disturbance torque about the y-axis versus time")
        # plt.savefig("Figures/linear_Tc_q")
        plt.show()

        plt.figure(3)
        plt.rcParams.update({"font.size": 16})
        plt.yscale("log")
        plt.plot(self.t, T_d[2, :], label="Total disturbance")
        # if T_GG.any() != None:
        #     plt.plot(self.t, T_GG[2,:], label="Gravity gradient")
        # if T_SRP.any() != None:
        #     plt.plot(self.t, T_SRP[2,:], label = "Solar radiation pressure")
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Disturbance torques about z-axis [Nm]")
        plt.title("Disturbance torque about the z-axis versus time")
        # plt.savefig("Figures/linear_Tc_q")
        plt.show()

        plt.figure(4)
        plt.rcParams.update({"font.size": 16})
        plt.yscale("log")
        plt.plot(self.t, T_d_mag, label="Total disturbance")
        # if T_GG.any() != None:
        #     plt.plot(self.t, T_GG_mag, label="Gravity gradient")
        # if T_SRP.any() != None:
        #     plt.plot(self.t, T_SRP_mag, label = "Solar radiation pressure")
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Disturbance torque magnitudes [Nm]")
        plt.title("Disturbance torque magnitudes versus time")
        # plt.savefig("Figures/linear_Tc_q")
        plt.show()

    def control_torque_versus_time(self, T_c):

        T_c_mag = np.sqrt(np.sum(T_c**2, axis=0))

        plt.figure(1)
        plt.rcParams.update({"font.size": 16})
        plt.yscale("log")
        plt.plot(self.t, T_c[0, :])
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Control torques about x-axis [Nm]")
        plt.title("Control torque about the x-axis versus time")
        # plt.savefig("Figures/linear_Tc_q")
        plt.show()

        plt.figure(2)
        plt.rcParams.update({"font.size": 16})
        plt.yscale("log")
        plt.plot(self.t, T_c[1, :])
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Control torques about y-axis [Nm]")
        plt.title("Control torque about the y-axis versus time")
        # plt.savefig("Figures/linear_Tc_q")
        plt.show()

        plt.figure(3)
        plt.rcParams.update({"font.size": 16})
        plt.yscale("log")
        plt.plot(self.t, T_c[2, :])
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Control torques about z-axis [Nm]")
        plt.title("Control torque about the z-axis versus time")
        # plt.savefig("Figures/linear_Tc_q")
        plt.show()

        plt.figure(4)
        plt.rcParams.update({"font.size": 16})
        plt.yscale("log")
        plt.plot(self.t, T_c_mag)
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Control torque magnitudes [Nm]")
        plt.title("Control torque magnitudes versus time")
        # plt.savefig("Figures/linear_Tc_q")
        plt.show()

    def omega_versus_time(self, omega_array):

        omega_mag = np.sqrt(np.sum(omega_array**2, axis=0))

        plt.figure(1)
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(self.t, omega_array[0, :])
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Angular velocity about the x-axis [-]")
        plt.title("Angular velocity about the x-axis versus time")
        plt.legend()
        # plt.savefig("Figures/linear_q_1.png")
        plt.show()

        plt.figure(2)
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(self.t, omega_array[1, :])
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Angular velocity about the y-axis [-]")
        plt.title("Angular velocity about the y-axis versus time")
        plt.legend()
        # plt.savefig("Figures/linear_q_1.png")
        plt.show()

        plt.figure(3)
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(self.t, omega_array[2, :])
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Angular velocity about the z-axis [-]")
        plt.title("Angular velocity about the z-axis versus time")
        plt.legend()
        # plt.savefig("Figures/linear_q_1.png")
        plt.show()

        plt.figure(4)
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({"font.size": 16})
        plt.plot(self.t, omega_mag)
        plt.xlabel("Time since epoch [s]")
        plt.ylabel("Magnitude of the angular velocity [-]")
        plt.title("Magnitude of the angular velocity versus time")
        plt.legend()
        # plt.savefig("Figures/linear_q_1.png")
        plt.show()


class thruster_allocation:
    """
    A class to solve the thruster allocation problem, aimed at determining optimal thrust configurations
    for given control torques in spacecraft or similar applications. The class handles the computation
    of feasible thrust vectors that satisfy given control torque requirements while adhering to specified
    torque limits and sensitivity settings.

    Attributes:
        T_c (numpy.ndarray): An array of controlled torque values set by the constructor after applying
                             limits and sensitivity adjustments.

    Methods:
        thruster_axes(phi, lam): Computes the directional axes of a thruster based on its orientation angles.
        thruster_configuration(array_thrusters): Calculates and prints the optimal thrust configuration
                                                 for a given array of thrusters.

    Parameters:
        control_torque (list or numpy.ndarray): The initial desired control torques obtained from the controller output.
        lower_limit_torque (list or numpy.ndarray): The minimum torques on each axis recognized as significant.
        k4 (float): A sensitivity parameter that adjusts how the torque thresholds are applied to the control torques.

    Usage:
        >>> ta = thruster_allocation([10, 15, 20], [2, 3, 3], 0.5)
        >>> ta.thruster_configuration([[1, 0, 0, 45, 30, 10], [0, 1, 0, 60, 45, 15], [0, 0, 1, 90, 90, 20]])
    """

    def __init__(self, control_torque, lower_limit_torque, k4):
        """
        Initializes the thruster_allocation with control torque thresholds and sensitivity settings.
        Processes each element of the control torque to adjust it based on the given lower limits and sensitivity.

        Parameters:
            control_torque (list or numpy.ndarray): Desired control torques from controller output.
            lower_limit_torque (list or numpy.ndarray): Threshold below which torques are adjusted or set to zero.
            k4 (float): Sensitivity factor that adjusts the torque threshold application.
        """
        self.T_c = np.empty(len(control_torque))
        for i in range(len(control_torque)):
            if control_torque[i] >= lower_limit_torque[i]:
                self.T_c[i] = control_torque[i]
            else:
                if control_torque[i] < k4 * lower_limit_torque[i]:
                    self.T_c[i] = 0
                else:
                    self.T_c[i] = lower_limit_torque[i]

    def thruster_axes(self, phi, lam):
        """
        Calculates the directional axes of the thruster based on its orientation angles.

        Parameters:
            phi (float): Rotation from the z-axis around the y-axis in the xz plane, in degrees.
            lam (float): Angle between the xz plane and the thrust vector, in degrees.

        Returns:
            numpy.ndarray: The normalized directional vector of the thruster.
        """
        return np.array(
            [
                np.cos(lam) * np.cos(np.pi / 2 - phi),
                np.sin(lam),
                np.cos(lam) * np.cos(phi),
            ]
        )

    def thruster_configuration(self, array_thrusters):
        """
        Computes and outputs the optimal thrust configuration for a specified array of thrusters based on the
        control torques set in the class. It calculates feasible thrust vectors that meet the torque requirements
        and adhere to thrust limits, printing the result.

        Parameters:
            array_thrusters (list of lists): Each element is a list containing the parameters [x, y, z, phi, lam, lim_value]
                                             of a thruster, where `lim_value` is the maximum thrust the thruster can exert.

        Outputs:
            Prints the optimal thrust vector and the minimal force magnitude required to achieve the control torques.
        """
        num_thrusters = len(array_thrusters)
        lim_values = [array_thrusters[i][-1] for i in range(num_thrusters)]
        thrust_limit = (
            np.ones(num_thrusters) * lim_values
        )  # Assuming equal limits for each thruster
        d = np.zeros((3, num_thrusters))
        e = np.zeros((3, num_thrusters))

        for i in range(num_thrusters):
            x, y, z, phi, lam, lim = array_thrusters[i]
            d[:, i] = [x, y, z]
            e[:, i] = self.thruster_axes(np.deg2rad(phi), np.deg2rad(lam))

        A_L = np.column_stack(
            [np.cross(d[:, i], e[:, i]) for i in range(num_thrusters)]
        )
        kern_A_L = null_space(A_L)

        # Error handling for the kernel
        valid_kernels = [
            kern_A_L[:, i]
            for i in range(kern_A_L.shape[1])
            if np.all(kern_A_L[:, i] > 0)
        ]

        if not valid_kernels:
            print("No valid kernel found. All kernels have non-positive elements.")
            return

        # Compute control actions if valid kernels exist
        pseudo_A_L = np.linalg.pinv(A_L)
        a_L = -pseudo_A_L @ self.T_c

        min_force = np.inf
        best_F_LUMIO = None

        for j in range(len(valid_kernels)):
            gamma_list = [
                thrust_limit[i] - a_L[i] / valid_kernels[j][i] for i in range(len(a_L))
            ]
            gamma_L = min(gamma_list)
            F_LUMIO = pseudo_A_L @ self.T_c + (gamma_L * valid_kernels[j])
            force_magnitude = np.linalg.norm(F_LUMIO)

            if force_magnitude < min_force:
                min_force = force_magnitude
                best_F_LUMIO = F_LUMIO

        if np.any(best_F_LUMIO < 0):
            print(
                "Error: Negative thrust instance detected in the optimal force vector."
            )
        else:
            print("Minimum force vector:", best_F_LUMIO)

