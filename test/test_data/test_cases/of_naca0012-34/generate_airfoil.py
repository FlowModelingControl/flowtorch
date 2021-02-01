#!/usr/bin/python
"""Script to generate modified 4 digit NACA airfoil geometries.

For a detailed description, formulas, and coefficients refer to:
http://www.aerospaceweb.org/question/airfoils/q0041.shtml

The script assumes a chord length of unity. The digits in the
airfoil's name have the following definition:

NACA 0012-34

digit | variable | meaning
------|----------|--------
   1  |     m    | maximum chamber relative to the chord length
   2  |     p    | position of the maximum chamber in tenth of the chord length
  3/4 |     t    | maximum airfoil thickness relative to the chord length
   5  |     -    | roundness if the nose; lower values indicate a sharper nose
   6  |     -    | location of the maximum thickness in thenth of the chord length
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

modification_parameters = {
    "34": [0.148450, 0.193233, -0.558166, 0.283208, 0.002, 0.315, -0.233333, -0.032407]
}


def compute_chamber(x, m, p):
    """Compute the airfoil's chamber.

    Parameters
    ----------
    x - array-like: points on the chrord for which to compute the chamber
    m - float: maximum chamber relative to the chord length
    p - float: position of the maximum chamber in tenth of the chord length

    Returns
    -------
    y_c - array-like: y-coordinate of the chamber

    """
    if p > 0:
        y_c = np.where(
            x < p,
            m / p**2 * (2*p*x - np.square(x)),
            m / (1-p)**2 * ((1-2*p) + 2*p*x - np.square(x))
        )
    else:
        y_c = np.zeros_like(x)
    return y_c


def compute_thickness(x, t):
    """Compute the airfoil's thickness.

    Parameters
    ----------
    x - array-like: points on the chord for which to compute the tickness
    t - float: maximum airfoil thickness relative to the chord length

    Returns
    -------
    y_t - array-like: airfoil thickness

    """
    y_t = t / 0.2 * (
        0.2969*np.sqrt(x) - 0.1260*x - 0.3516*np.square(x) +
        0.2843*np.power(x, 3) - 0.1015*np.power(x, 4)
    )
    return y_t


def compute_modified_thickness(x, t, mod):
    """Compute the airfoil's thickness for 'modified' profiles.

    Parameters
    ----------
    x - array-like: points on the chord for which to compute the tickness
    t - float: maximum airfoil thickness relative to the chord length
    mod - String: two digits describing the modification
        (nose roundness and shift of maximum thickness)

    Returns
    -------
    y_t - array-like: modified airfoil thickness

    """
    print("Computing airfoil thickness for modification {:s}".format(mod))

    if mod in modification_parameters.keys():
        a_0, a_1, a_2, a_3 = modification_parameters[mod][:4]
        d_0, d_1, d_2, d_3 = modification_parameters[mod][4:]
    else:
        raise ValueError

    y_t_ahead = t/0.2 * (
        a_0*np.sqrt(x) + a_1*x + a_2*np.square(x) + a_3*np.power(x, 3)
    )
    y_t_aft = t/0.2 * (
        d_0 + d_1*(1-x) + d_2*np.square(1-x) + d_3*np.power(1-x, 3)
    )
    y_t_max = np.max(y_t_ahead)
    max_loc = np.argmax(y_t_ahead)

    print("Maximum airfoil thickness of t_max={:2.2f} at x={:2.2f}".format(
        2*y_t_max, x[max_loc]))

    return np.concatenate((y_t_ahead[:max_loc], y_t_aft[max_loc:]))


def compute_theta(x, y_c):
    """Compute angle between chord line and tangent at chamber.
    This function uses finite differences rather than the analytical
    expression to compute dy/dx. The advantages is that all profiles
    can be treated the same regardless of modifications. Moreover,
    no cases for different values of p and x have to be distinguished.

    Parameters
    ----------
    x - array-like: points on the chord for which to compute the tickness
    y_c - array-like: airfoil chamber cooresponding to x

    Returns
    -------
    theta - array-like: angle between coord line and tangents at chamber

    """
    if not x.shape == y_c.shape:
        raise ValueError("Error: x and y_c must have the same dimension.")
    dx = x[1:] - x[:-1]
    dy = y_c[1:] - y_c[:-1]
    dx = np.concatenate((dx, np.array([dx[-1]])))
    dy = np.concatenate((dy, np.array([dy[-1]])))
    return np.arctan(dy/dx)


def generate_profile(m, p, t, mod=None, n_points=200):
    if mod is None:
        ext = ""
    else:
        ext = "-" + mod

    x = np.linspace(0.0, np.pi, n_points)
    x = (1.0 - np.cos(x)) * 0.5
    if mod is None:
        y_t = compute_thickness(x, t)
    else:
        y_t = compute_modified_thickness(x, t, mod)

    y_c = compute_chamber(x, m, p)
    theta = compute_theta(x, y_c)

    x_final = np.concatenate(
        (x - y_t*np.sin(theta), x[::-1]+y_t[::-1]*np.sin(theta[::-1]))
    )
    y_final = np.concatenate(
        (y_c + y_t*np.cos(theta), y_c[::-1] - y_t[::-1]*np.cos(theta[::-1]))
    )

    return x_final, y_final


def vector_to_string(vector, sep):
    """Convert 1D vector to String.
    Parameters
    ----------
    vector - array-like : vector to convert
    sep - String : String to use as separator between elements
    Returns
    -------
    strVector - String : vector as String with elements separated by sep
    """
    strVector = ""
    for el in vector[:-1]:
        strVector += str(el) + sep
    return strVector + str(vector[-1])


class Triangle():
    """Helper class to write STL files.
    """

    def __init__(self, p1, p2, p3):
        """Create a Triangle object based on three points.
        Parameters
        ----------
        p1, p2, p3 - array-like : triangle vertices of length 2 - (x, y)
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def points_to_string(self, prefix):
        """Return points as String.
        Parameters
        ----------
        prefix - String : prefix to add before each point
        Returns
        -------
        points as String
        """
        p1str = prefix + vector_to_string(self.p1, " ")
        p2str = prefix + vector_to_string(self.p2, " ")
        p3str = prefix + vector_to_string(self.p3, " ")
        return p1str + "\n" + p2str + "\n" + p3str

    def normal_to_string(self):
        """Compute vector normal to triangle with unit length."""
        normal = np.cross(self.p3-self.p1, self.p2-self.p1)
        normal = normal / np.linalg.norm(normal)
        return vector_to_string(normal, " ")


def export_stl(x, y, depth, region, file_name):
    """Generate STL geometry file based on points given as (x,y).
    The function expects the points to be ordered as follows:
    - points on the upper surface ordered according to x (increasing)
    - points on the lower surface ordered according to x (decreasing)

    Parameters
    ----------
    x, y - array-like: points describing the airfoil's contour
    depth - float: profile depth in z-direction
    region - String: name of the airoil STL region
    file_name - String: name of the STL file (including extension)

    """
    triangles = []
    for t in range(len(x)-1):
        # upper triangle
        triangles.append(Triangle(np.asarray([x[t], y[t], 0.5*depth]),
                                  np.asarray([x[t+1], y[t+1], -0.5*depth]),
                                  np.asarray([x[t], y[t], -0.5*depth]))
                         )
        # lower triangle
        triangles.append(Triangle(np.asarray([x[t], y[t], 0.5*depth]),
                                  np.asarray([x[t+1], y[t+1], 0.5*depth]),
                                  np.asarray([x[t+1], y[t+1], -0.5*depth]))
                         )

    stl_file = open(file_name, 'w')
    stl_file.write("solid " + region + '\n')
    for triangle in triangles:
        stl_file.write("facet normal " + triangle.normal_to_string() + "\n")
        stl_file.write("    outer loop\n")
        prefix = ' ' * 8 + "vertex "
        stl_file.write(triangle.points_to_string(prefix) + "\n")
        stl_file.write("    endloop\n")
        stl_file.write("endfacet\n")
    stl_file.write("endsolid " + region)
    stl_file.close()


def plot_airfoil(x, y, airfoil_name, reference_data=None):
    plt.plot(x, y, label=airfoil_name)
    if not reference_data is None:
        ref_data = pd.read_csv(reference_data, sep="     ",
                            skiprows=1, names=["x", "y"], engine="python")
        plt.scatter(ref_data.x, ref_data.y, marker="x", c="C1", label="reference data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(airfoil_name)
    plt.savefig(airfoil_name + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    x, y = generate_profile(0, 0, 0.12, "34", 200)
    export_stl(x, y, 0.2, "airfoil", "naca0012-34.stl")
    plot_airfoil(x, y, "NACA-0012-34", "naca0012-34.csv")
