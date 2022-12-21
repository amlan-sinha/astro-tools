"""
The following code is adapted from "pybeeastro" written by Ryne Beeson.
"""

class SpaceBody:

    __author__ = 'rynebeeson'

    G = 6.67384E-20  # (km ** 3) / (kg * s ** 2)
    mass = NotImplemented
    mu = NotImplemented
    mu_of_parent = NotImplemented
    semimajor_axis = NotImplemented
    eccentricity = NotImplemented
    inclination = NotImplemented
    Omega = NotImplemented
    omega = NotImplemented
    true_anomaly = NotImplemented
    hills_radius = NotImplemented
    equatorial_radius = NotImplemented
    PI = 3.14159265
    TAU = 2.0 * PI

    def get_orbital_period(self):
        from numpy import sqrt
        return self.TAU * sqrt(self.semimajor_axis ** 3 / self.mu_of_parent)

    def _set_hills_radius(self):
        self.hills_radius = self.semimajor_axis * (self.mu / (3.0 * self.mu_of_parent))**(1/3.0)

    def _set_mu_for_children(self):
        self.mu = self.G * self.mass

class Sun(SpaceBody):

    def __init__(self):
        self.mass = 1.989E30  # (kg)


class Earth(SpaceBody):

    standard_gravity = 9.80665  # (m / s^2)

    def __init__(self):
        sun = Sun()
        self.mass = 5.97219E24  # (kg)
        self._set_mu_for_children()
        self.mu_of_parent = self.G * sun.mass
        self.semimajor_axis = 1.495978E8  # (km)
        self.eccentricity = 0.0167
        self.inclination = 0.0  # rad
        self._set_hills_radius()
        self.equatorial_radius = 6371.0  # (km)


class Moon(SpaceBody):

    def __init__(self):
        earth = Earth()
        self.mass = 7.3477E22  # (kg)
        self._set_mu_for_children()
        self.mu_of_parent = self.G * earth.mass
        self.semimajor_axis = 3.844E5  # (km)
        self.eccentricity = 0.0549
        self.inclination = 0.0898844564  # rad
        self._set_hills_radius()
        self.equatorial_radius = 1737.0