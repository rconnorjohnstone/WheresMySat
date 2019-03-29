#!/usr/bin/env python
"""Propagator Module

Author: Hunter Mellema
Summary: Provides a forces model system for
"""
# Standard library imports
from scipy.integrate import solve_ivp
from math import sqrt, exp

# third party imports
from numba import jit

### CONSTANTS ####
from WheresMySat import mu_Ea, J2_Ea

### Propagator
def propagate_sc_traj(istates, force_model, times, dt=0.01):
    """Uses the lsoda"""
    states = [istates]
    last_time = times[0]
    for time_next in times[1:]:
        sol = solve_ivp(force_model.ode, [last_time, time_next],
                        states[-1], method="LSODA", atol=1e-8, rtol=1e-6)

        sol_val = [y[len(sol.t)-1] for y in sol.y]
        states.append(sol_val)
        last_time = time_next


    return states

### Forces
class ForceModel(object):
    """ Defines a force model to use for integration of trajectories or
    stms

    Args:
        force_list (list[functions]): list of force functions to use for
            model

    """
    def __init__(self, force_list):
        self.force_list = force_list


    def __call__(self, t, state_vec):
        """
        """
        xddot, yddot, zddot = map(sum, zip(*[fxn(state_vec) for
                                             fxn in self.force_list]))

        out_state = [state_vec[3], state_vec[4], state_vec[5],
                     xddot, yddot, zddot]

        for i, _ in enumerate(state_vec[6:]):
            out_state.append(0)

        return out_state


# Forces you can add to the force model
def point_mass(state_vec):
        """Calculates the x, y, z accelerations due to point
            mass gravity model

        """
        mu = set_mu(state_vec)
        x, y, z = state_vec[0:3]
        r = norm(state_vec[0:3])

        return  [-mu * coord / r**3 for coord in state_vec[0:3]]

def set_mu(state_vec):
    """ """
    mu = state_vec[6] if 6 < len(state_vec) else mu_Ea

    return mu

def j2_accel(state_vec):
        """Calculates the J2 x, y, z accelerations

        """
        j2 = set_j2(state_vec)
        x, y, z = state_vec[0:3]
        r = norm(state_vec[0:3])
        xddot = - 3 * j2 * x / (2 * r**5) * (1 - 5 * z**2 / r**2)
        yddot = - 3 * j2 * y / (2 * r**5) * (1 - 5 * z**2 / r**2)
        zddot = - 3 * j2 * z / (2 * r**5) * (3 - 5 * z**2 / r**2)

        return [xddot, yddot, zddot]

def set_j2(state_vec):
    """"""
    j2 = state_vec[7] if 7 < len(state_vec) else J2_Ea

    return j2

@jit
def norm(vec):
    """ Computes the 2 norm of a vector or vector slice """
    return sqrt(sum([i**2 for i in vec]))
