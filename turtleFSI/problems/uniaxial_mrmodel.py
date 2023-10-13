# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
This file is a problem file for the uniaxial tension test of a Mooney-Rivlin material.
"""

from pathlib import Path

from dolfin import HDF5File, Mesh, MeshFunction, AutoSubDomain, DirichletBC, near \
    , interpolate, Expression, VectorFunctionSpace, assemble, inner, Measure

from turtleFSI.modules.common import Piola1, E
from turtleFSI.problems import *

parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 

def set_problem_parameters(default_variables, **namespace):

    E_s_val = 1E6
    nu_s_val = 0.45
    mu_s_val = E_s_val/(2*(1+nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val*2.*mu_s_val/(1. - 2.*nu_s_val)
    
    default_variables.update(dict(
        # Temporal variables
        T=625,     # End time [s] based on 20 micrometer per second, this should be equivalent to 1.5 stretch
        dt=1,       # Time step [s]
        checkpoint_step=1000, # Checkpoint frequency
        theta=0.5,     # Temporal scheme
        save_step=100,

        solid_properties={"dx_s_id":1,"material_model":"MooneyRivlin","rho_s":1.0E3,"mu_s":mu_s_val,"lambda_s":lambda_s_val,"C01":0.02e6,"C10":0.0,"C11":1.8e6},
        gravity=None,   # Gravitational force [m/s**2]

        # Problem specific
        dx_s_id=1,     # Id of the solid domain
        folder="uniaxial_mrmodel_results",          # Folder to store the results
        fluid="no_fluid",                 # Do not solve for the fluid
        extrapolation="no_extrapolation",  # No displacement to extrapolate
        speed = 0.000002, # 20 micrometer per second
        stress_list = [], # List to store the stress
        strain_list = []
    ))

    return default_variables


def get_mesh_domain_and_boundaries(**namespace):
    # Read mesh / here I assume mesh folder is parallel to the problem folder
    mesh_folder = Path(__file__).parent.parent / "mesh" / "uniaxial"
    mesh_path = mesh_folder / "tissue.h5"
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), str(mesh_path), "r")
    hdf.read(mesh, "/mesh", False)

    # Mark boundaries
    left_wall = AutoSubDomain(lambda x: near(x[0], 0))
    right_wall = AutoSubDomain(lambda x: near(x[0], 0.005))

    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)

    boundaries.set_all(0)

    left_wall.mark(boundaries, 1)
    right_wall.mark(boundaries, 2)

    # Mark domain
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries,


def create_bcs(DVP, speed, boundaries, **namespace):
    # Move left and wall in y direction / constrained in x and z
    # Give DirichletBC to the displacement and velocity
    bcs = []
    d_left_y = DirichletBC(DVP.sub(0).sub(1), ((0.0)), boundaries, 1)
    d_left_z = DirichletBC(DVP.sub(0).sub(2), ((0.0)), boundaries, 1)
    u_left_y = DirichletBC(DVP.sub(1).sub(1), ((0.0)), boundaries, 1)
    u_left_z = DirichletBC(DVP.sub(1).sub(2), ((0.0)), boundaries, 1)
    bcs.append(d_left_y)
    bcs.append(d_left_z)
    bcs.append(u_left_y)
    bcs.append(u_left_z)

    d_right_y = DirichletBC(DVP.sub(0).sub(1), ((0.0)), boundaries, 2)
    d_right_z = DirichletBC(DVP.sub(0).sub(2), ((0.0)), boundaries, 2)
    u_right_y = DirichletBC(DVP.sub(1).sub(1), ((0.0)), boundaries, 2)
    u_right_z = DirichletBC(DVP.sub(1).sub(2), ((0.0)), boundaries, 2)
    bcs.append(d_right_y)
    bcs.append(d_right_z)
    bcs.append(u_right_y)
    bcs.append(u_right_z)

    
    # We stretch the left and right wall in x direction with constant velocity
    u_left_x = DirichletBC(DVP.sub(1).sub(0), ((-1 * speed)), boundaries, 1)
    u_right_x = DirichletBC(DVP.sub(1).sub(0), ((speed)), boundaries, 2)

    bcs.append(u_left_x)
    bcs.append(u_right_x)

    return dict(bcs=bcs)


def post_solve(dvp_, solid_properties, mesh, stress_list, strain_list, dx_s, **namespace):

    V_f = VectorFunctionSpace(mesh, "CG", 1)
    x_vector = interpolate(Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
    pk1_stress = Piola1(dvp_["n"].sub(0), solid_properties[0])
    volume_averaged_pk1_stress = assemble(inner(x_vector, pk1_stress * x_vector) * dx_s[0]) / assemble(inner(x_vector, x_vector) * dx_s[0])
    stress_list.append(volume_averaged_pk1_stress)
    print("Volume averaged stress: ", volume_averaged_pk1_stress)

    # Compute Green-Lagrange strain
    green_lagrange_strain = E(dvp_["n"].sub(0))
    volume_averaged_green_lagrange_strain = assemble(inner(x_vector, green_lagrange_strain * x_vector) * dx_s[0]) / assemble(inner(x_vector, x_vector) * dx_s[0])
    print("Volume averaged strain: ", volume_averaged_green_lagrange_strain)
    strain_list.append(volume_averaged_green_lagrange_strain)
    return dict(stress_list=stress_list, strain_list=strain_list)

def finished(stress_list, strain_list, **namespace):
    # Add one to strain list to make it stretch
    strech_list = [x+1 for x in strain_list]
    # convert stress from Pa to MPa
    stress_list = [x/1e6 for x in stress_list]
    # plot the stress-strain curve
    import matplotlib.pyplot as plt
    plt.plot(strech_list, stress_list)
    plt.xlabel("Stretch")
    plt.ylabel("Stress [MPa]")
    plt.savefig("stress_strain_curve.png")

    



