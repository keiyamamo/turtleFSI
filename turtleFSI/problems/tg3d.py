import pickle
from os import path
import sys
import numpy as np

from dolfin import *
from turtleFSI.problems import *

"""
Taylor-Green vortex in 2D with fixed domain
This problem can be used to test the accuracy of the fluid solver
The mesh can be either structured or unstructured based on the user's choice and availability of pygmsh
"""

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
_compiler_parameters = dict(parameters["form_compiler"])

# Override some problem specific parameters
def set_problem_parameters(default_variables, **namespace):
    default_variables.update(dict(
        mu_f=1/1600,                        # dynamic viscosity of fluid, 0.01 as kinematic viscosity
        T=20,
        dt=0.002,
        theta=0.5,                        # Crank-Nicolson
        rho_f = 1,                        # density of fluid
        folder="tg3d_results",
        solid = "no_solid",               # no solid
        extrapolation="no_extrapolation", # no extrapolation since the domain is fixed
        save_step=5,
        checkpoint_step=500,
        v_deg=2,
        p_deg=1,
        d_deg=1,
        atol=1e-8,
        rtol=1e-8,
        N=10,                              # number of points along x or y axis when creating structured mesh
        recompute=100,
        recompute_tstep=100,
        constrained_domain=PeriodicDomain(),
        save_deg=2,
        kinematic_energy_previous = 0,
        compiler_parameters=_compiler_parameters,
        ))

    return default_variables


def near(x, y, tol=1e-12):
    return bool(abs(x - y) < tol)


class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        return bool((near(x[0], -pi) or near(x[1], -pi) or near(x[2], -pi)) and
                    (not (near(x[0], pi) or near(x[1], pi) or near(x[2], pi))) and on_boundary)

    def map(self, x, y):
        if near(x[0], pi) and near(x[1], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0 * pi
            y[1] = x[1] - 2.0 * pi
            y[2] = x[2] - 2.0 * pi
        elif near(x[0], pi) and near(x[1], pi):
            y[0] = x[0] - 2.0 * pi
            y[1] = x[1] - 2.0 * pi
            y[2] = x[2]
        elif near(x[1], pi) and near(x[2], pi):
            y[0] = x[0]
            y[1] = x[1] - 2.0 * pi
            y[2] = x[2] - 2.0 * pi
        elif near(x[1], pi):
            y[0] = x[0]
            y[1] = x[1] - 2.0 * pi
            y[2] = x[2]
        elif near(x[0], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0 * pi
            y[1] = x[1]
            y[2] = x[2] - 2.0 * pi
        elif near(x[0], pi):
            y[0] = x[0] - 2.0 * pi
            y[1] = x[1]
            y[2] = x[2]
        else:  # near(x[2], pi):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - 2.0 * pi

def get_mesh_domain_and_boundaries(N, **namespace):

    mesh = BoxMesh(Point(-pi, -pi, -pi), Point(pi, pi, pi), N, N, N)
      
    # Mark the boundaries
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries

class analytical_velocity(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def eval(self, value, x):
        value[0] = sin(x[0])*cos(x[1])*cos(x[2])
        value[1] = -cos(x[0])*sin(x[1])*cos(x[2])
        value[2] = 0

    def value_shape(self):
        return (3,)

class analytical_pressure(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, value, x):
        value[0] = 1./16.*(cos(2*x[0])+cos(2*x[1]))*(cos(2*x[2])+2)
    
    def value_shape(self):
        return ()

def create_bcs(**namespace):
    """
    empty boundary conditions
    """
    bcs = []
    
    return dict(bcs=bcs)
    
def initiate(dvp_, DVP, **namespace):
    """
    Initialize solution using analytical solution.
    """
    inital_velocity = analytical_velocity()
    inital_pressure = analytical_pressure()
    # generate functions of the initial solution from expressions
    ui = interpolate(inital_velocity, DVP.sub(1).collapse())
    pi = interpolate(inital_pressure, DVP.sub(2).collapse())
    # assign the initial solution to dvp_
    assign(dvp_["n"].sub(1), ui)
    assign(dvp_["n-1"].sub(1), ui)
    assign(dvp_["n"].sub(2), pi)
    assign(dvp_["n-1"].sub(2), pi)

    kinetic_energy_list = []
    dissipation_list = []
    enstrophy_list = []

    time_list = []

    return dict(dvp_=dvp_, kinetic_energy_list=kinetic_energy_list, dissipation_list=dissipation_list, 
                time_list=time_list, enstrophy_list=enstrophy_list)


def post_solve(DVP, t, dt, dvp_, counter, kinetic_energy_list, dissipation_list, kinematic_energy_previous, time_list, enstrophy_list, **namespace):
    """
    Compute errors after solving 
    """
    # Get velocity, and pressure
    if counter % 5 == 0:
        v = dvp_["n"].sub(1, deepcopy=True)
        
        kinetic = assemble(0.5 * dot(v, v) * dx) / (2 * pi)**3
        kinetic_dissipation_rate = -(kinetic - kinematic_energy_previous) / (dt * 5)
        enstrophy = assemble(0.5 * dot(curl(v), curl(v)) * dx) / (2 * pi)**3

        kinetic_energy_list.append(kinetic)
        dissipation_list.append(kinetic_dissipation_rate)
        enstrophy_list.append(enstrophy)
        time_list.append(t)


        # print info
        if MPI.rank(MPI.comm_world) == 0:
            print("Kinetic energy: ", kinetic)
            print("Dissipation rate: ", kinetic_dissipation_rate)
            print("Enstrophy: ", enstrophy)

        return dict(kinetic_energy_list=kinetic_energy_list, dissipation_list=dissipation_list, 
                    kinematic_energy_previous=kinetic, time_list=time_list, enstrophy_list=enstrophy_list)


def finished(kinetic_energy_list, dissipation_list, time_list, enstrophy_list, results_folder, **namespace):

    if MPI.rank(MPI.comm_world) == 0:
        np.savetxt(path.join(results_folder, 'Time.txt'), time_list, delimiter=',')
        np.savetxt(path.join(results_folder, 'KineticEnergy.txt'), kinetic_energy_list, delimiter=',')
        np.savetxt(path.join(results_folder, 'DissipationRate.txt'), dissipation_list, delimiter=',')
        np.savetxt(path.join(results_folder, 'Enstrophy.txt'), enstrophy_list, delimiter=',')
        
        


