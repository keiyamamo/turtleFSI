# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""Problem file for running the "CSM" benchmarks in [1]. The problem is beam under load.

[1] Turek, Stefan, and Jaroslav Hron. "Proposal for numerical benchmarking of fluid-structure interaction
between an elastic object and laminar incompressible flow." Fluid-structure interaction.
Springer, Berlin, Heidelberg, 2006. 371-385."""

from dolfin import *
import numpy as np
from os import path
from mpi4py import MPI as pyMPI

from turtleFSI.problems import *


def set_problem_parameters(default_variables, **namespace):
    # Parameters
    E_s_val = 1.0e3
    nu_s_val = 0.3
    mu_s_val = E_s_val/(2*(1 + nu_s_val))
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    default_variables.update(dict(
        # Temporal variables
        T=4.0,          # End time [s]
        dt=0.01,       # Time step [s]
        theta=0.51,     # Temporal scheme

        # Physical constants
        rho_s=1.0,   # Solid density[kg/m3]
        mu_s=mu_s_val,  # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,  # Solid Poisson ratio [-]
        gravity=0.0,   # Gravitational force [m/s^2]
        lambda_s=lambda_s_val,  # Solid 1rst Lamé coef. [Pa]

        # Problem specific
        dx_f_id=0,     # Id of the fluid domain
        dx_s_id=1,     # Id of the solid domain
        folder="beam_result",          # Folder to store the results
        fluid="no_fluid",                 # Do not solve for the fluid
        extrapolation="no_extrapolation",  # No displacement to extrapolate
        recompute_tstep=10,
        recompute=10,
        ))

    return default_variables


# Sub domain for clamp at left end
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return near(x[0], 1.) and on_boundary


def get_mesh_domain_and_boundaries(**namespace):
    # Read mesh
    mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 60, 10, 5)
    
    # Mark boundaries
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    force_boundary = AutoSubDomain(right)
    force_boundary.mark(boundaries, 3)

    clamped_boundary = AutoSubDomain(left)
    clamped_boundary.mark(boundaries, 1)

    # Mark domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries


def initiate(**namespace):
    # Lists to hold results
    displacement_y_list = []
    time_list = []

    kinematic_energy_list = []
    elastic_energy_list = []
    damping_list = []
    total_energy_list = []


    return dict(displacement_y_list=displacement_y_list, time_list=time_list,
                kinematic_energy_list=kinematic_energy_list, elastic_energy_list=elastic_energy_list,
                damping_list=damping_list, total_energy_list=total_energy_list)


def create_bcs(DVP, boundaries, psi, F_solid_linear, T, **namespace):
    # Clamp on the left hand side
    u_left = DirichletBC(DVP.sub(0), ((0.0, 0., 0.0)), boundaries, 1)
    v_left = DirichletBC(DVP.sub(1), ((0.0, 0., 0.0)), boundaries, 1)
    ds_right = Measure("ds", domain=boundaries.mesh(), subdomain_data=boundaries, subdomain_id=3)
    p0 = 1.0
    cutoff_Tc = T/5 
    p = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

    # Force on the right hand side
    F_solid_linear -= dot(p, psi)*ds_right

    return dict(bcs=[u_left, v_left], F_solid_linear=F_solid_linear, p=p)

################################################################################
# the function mpi4py_comm and peval are used to overcome FEniCS limitation of
# evaluating functions at a given mesh point in parallel.
# https://fenicsproject.discourse.group/t/problem-with-evaluation-at-a-point-in
# -parallel/1188


def mpi4py_comm(comm):
    '''Get mpi4py communicator'''
    try:
        return comm.tompi4py()
    except AttributeError:
        return comm


def peval(f, x):
    '''Parallel synced eval'''
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf*np.ones(f.value_shape())

    comm = mpi4py_comm(f.function_space().mesh().mpi_comm())
    yglob = np.zeros_like(yloc)
    comm.Allreduce(yloc, yglob, op=pyMPI.MIN)

    return yglob
################################################################################


def pre_solve(t, p, **namespace):
    # Update the pressure value
    p.t = t

    return dict(p=p)
    

def post_solve(t, dvp_, displacement_y_list, time_list, verbose, solid_properties,
               kinematic_energy_list, elastic_energy_list, damping_list, total_energy_list, **namespace):
    # Add time
    time_list.append(t)

    # Add displacement
    d = dvp_["n"].sub(0, deepcopy=True)
    u = dvp_["n"].sub(1, deepcopy=True)

    d_eval = peval(d, [1., 0.05, 0.])
    dsy = d_eval[1]

    displacement_y_list.append(dsy)

    if MPI.rank(MPI.comm_world) == 0 and verbose:

        print("Distance y: {:e}".format(dsy))


def finished(results_folder, displacement_y_list, time_list, **namespace):
    # Store results when the computation is finished
    if MPI.rank(MPI.comm_world) == 0:
        np.savetxt(path.join(results_folder, 'Time.txt'), time_list, delimiter=',')
        np.savetxt(path.join(results_folder, 'dis_y.txt'), displacement_y_list, delimiter=',')