# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
This module implements the monolithic Fluid-Structure Interaction (FSI) solver
used in the turtleFSI package.
"""

from dolfin import *
from pathlib import Path
import json
import time
from pprint import pprint

from turtleFSI.utils import *
from turtleFSI.problems import *

# Get user input
args = parse()

# Import the problem
if Path.cwd().joinpath(args.problem+'.py').is_file():
    exec("from {} import *".format(args.problem))
else:
    try:
        exec("from turtleFSI.problems.{} import *".format(args.problem))
    except ImportError:
        raise ImportError("""Can not find the problem file. Make sure that the
        problem file is specified in the current directory or in the solver
        turtleFSI/problems/... directory.""")

# Get problem specific parameters
default_variables.update(set_problem_parameters(**vars()))

# Update variables from commandline
for key, value in list(args.__dict__.items()):
    if value is None:
        args.__dict__.pop(key)

# If restart folder is given, read previous settings
default_variables.update(args.__dict__)
if default_variables["restart_folder"] is not None:
    restart_folder = Path(default_variables["restart_folder"])
    restart_folder = restart_folder if "Checkpoint" in restart_folder.__str__() else restart_folder.joinpath("Checkpoint")
    with open(restart_folder.joinpath("default_variables.json"), "r") as f:
        restart_dict = json.load(f)
    default_variables.update(restart_dict)
    default_variables["restart_folder"] = restart_folder

# Set variables in global namespace
vars().update(default_variables)

# Print out variables
if MPI.rank(MPI.comm_world) == 0 and verbose:
    pprint(default_variables)

# Create folders
vars().update(create_folders(**vars()))

# Get mesh information
mesh, domains, boundaries = get_mesh_domain_and_boundaries(**vars())

# Save mesh, domains, and boundaries for post-processing
if restart_folder is None:
    h5_mesh_path = results_folder.joinpath("Mesh", "mesh.h5")
    with HDF5File(mesh.mpi_comm(), h5_mesh_path.__str__(), "w") as hdf:
        hdf.write(mesh, "/mesh")
        hdf.write(boundaries, "/boundaries")
        hdf.write(domains, "/domains")

# Control FEniCS output
set_log_level(loglevel)

# Finite Elements for deformation (de), velocity (ve), and pressure (pe)
de = VectorElement('CG', mesh.ufl_cell(), d_deg)
ve = VectorElement('CG', mesh.ufl_cell(), v_deg)
pe = FiniteElement('CG', mesh.ufl_cell(), p_deg)

# Define coefficients
k = Constant(dt)
n = FacetNormal(mesh)

# Define function space
# When using a biharmonic mesh lifting operator, we have to add a fourth function space.
if extrapolation == "biharmonic":
    Elem = MixedElement([de, ve, pe, de])
else:
    Elem = MixedElement([de, ve, pe])

DVP = FunctionSpace(mesh, Elem)

# Create one function for time step n, n-1, and n-2
dvp_ = {}
d_ = {}
v_ = {}
p_ = {}
w_ = {}

times = ["n-2", "n-1", "n"]
for time_ in times:
    dvp = Function(DVP)
    dvp_[time_] = dvp
    dvp_list = split(dvp)

    d_[time_] = dvp_list[0]
    v_[time_] = dvp_list[1]
    p_[time_] = dvp_list[2]
    if extrapolation == "biharmonic":
        w_[time_] = dvp_list[3]

if extrapolation == "biharmonic":
    phi, psi, gamma, beta = TestFunctions(DVP)
else:
    phi, psi, gamma = TestFunctions(DVP)

# Differentials
ds = Measure("ds", subdomain_data=boundaries)
dS = Measure("dS", subdomain_data=boundaries)
dx = Measure("dx", subdomain_data=domains)

# Domains
exec("from turtleFSI.modules.domain import assign_domain_properties")
vars().update(assign_domain_properties(**vars()))
if MPI.rank(MPI.comm_world) == 0:
    print("{} solid region(s) found, using following parameters".format(len(dx_s_id_list)))
    for solid_region in solid_properties:
        print(solid_region)
    print("{} fluid region(s) found, using following parameters".format(len(dx_f_id_list)))
    for fluid_region in fluid_properties:
        print(fluid_region)

# Define solver
# Adding the Matrix() argument is a FEniCS 2018.1.0 hack
up_sol = LUSolver(Matrix(), linear_solver)

# Get variation formulations
exec("from turtleFSI.modules.{} import fluid_setup".format(fluid))
vars().update(fluid_setup(**vars()))
exec("from turtleFSI.modules.{} import solid_setup".format(solid))
vars().update(solid_setup(**vars()))
exec("from turtleFSI.modules.{} import extrapolate_setup".format(extrapolation))
vars().update(extrapolate_setup(**vars()))

# Any action before the simulation starts, e.g., initial conditions or overwriting parameters from restart
vars().update(initiate(**vars()))

# Create boundary conditions
vars().update(create_bcs(**vars()))

# Set up Newton solver
exec("from turtleFSI.modules.{} import solver_setup, newtonsolver".format(solver))
vars().update(solver_setup(**vars()))

# Functions for residuals
dvp_res = Function(DVP)
chi = TrialFunction(DVP)

# Set initial conditions from restart folder
if restart_folder is not None:
    start_from_checkpoint(**vars())

timer = Timer("Total simulation time")
timer.start()
previous_t = 0.0
stop = False
first_step_num = counter # This is so that the solver will recompute the jacobian on the first step of the simulation
while t <= T + dt / 10 and not stop:  # + dt / 10 is a hack to ensure that we take the final time step t == T
    t += dt
    # Pre solve hook
    tmp_dict = pre_solve(**vars())
    if tmp_dict is not None:
        vars().update(tmp_dict)

    # Solve
    vars().update(newtonsolver(**vars()))
    # NOTE: To compute acceleration term inside post_solve function, 
    #       we need to place post_solve right after newtonsolver
    tmp_dict = post_solve(**vars())

    # Update vectors
    for i, t_tmp in enumerate(times[:-1]):
        dvp_[t_tmp].vector().zero()
        dvp_[t_tmp].vector().axpy(1, dvp_[times[i+1]].vector())

    # After solve hook
    if tmp_dict is not None:
        vars().update(tmp_dict)

    # Checkpoint
    if counter % checkpoint_step == 0:
        checkpoint(**vars())

    # Store results
    if counter % save_step == 0:
        vars().update(save_files_visualization(**vars()))

    # Update the time step counter
    counter += 1

    # Print time per time step
    if MPI.rank(MPI.comm_world) == 0:
        previous_t = print_information(**vars())

    # pause simulation if pauseturtle exists
    pauseturtle = check_if_pause(results_folder)
    while pauseturtle:
        time.sleep(5)
        pauseturtle = check_if_pause(results_folder)

    # stop simulation cleanly if killturtle exists
    killturtle = check_if_kill(results_folder, killtime, timer)
    if killturtle:
        checkpoint(**vars())
        stop = True

# Print total time
timer.stop()
if MPI.rank(MPI.comm_world) == 0:
    if verbose:
        print("Total simulation time {0:f}".format(timer.elapsed()[0]))
    else:
        print("\nTotal simulation time {0:f}".format(timer.elapsed()[0]))

# Merge visualization files
if restart_folder is not None and MPI.rank(MPI.comm_world) == 0:
    print("Merging visualization files")
    merge_visualization_files(**vars())

# Post-processing of simulation
finished(**vars())
