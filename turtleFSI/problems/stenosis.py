import numpy as np
import random
import os

from turtleFSI.problems import *
from dolfin import DirichletBC, cpp, MeshValueCollection, Expression, UserExpression, MeshFunction, MPI, File


# Set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet normals n('+') within interior boundaries (dS). 
# For 3D mesh the value should be "shared_vertex", for 2D mesh "shared_facet", the default value is "none".
parameters["ghost_mode"] = "shared_vertex" #3D case
_compiler_parameters = dict(parameters["form_compiler"])

def set_problem_parameters(default_variables, **namespace):
    
    default_variables.update(dict(
        # Temporal parameters
        T=15, # s
        dt=1e-4, # s
        theta = 0.5001, # Shifted-Crank-Nicolson

        # Fluid parameters
        Re=600,
        D=0.00635, # m
        # nu=0.0031078341013824886, # mm^2/ms #3.1078341E-6 m^2/s, #0.003372 Pa-s/1085 kg/m^3 this is nu_inf (m^2/s)
        mu_f =0.003372, # fluid dynamic viscosity (Pa-s)
        rho_f=1085,   # kg/m^3, density of fluid

        # Solid parameters
        solid = "no_solid",               # no solid
        extrapolation="no_extrapolation", # no extrapolation since the domain is fixed

        checkpoint_step=500,

        atol=1e-6, # Absolute tolerance in the Newton solver
        rtol=1e-6,# Relative tolerance in the Newton solver

        ))

    return default_variables


def get_mesh_domain_and_boundaries(**namespace):
    # Import mesh file
    mesh = Mesh()
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with XDMFFile(MPI.comm_world, "mesh/Stenosis_400K/mesh.xdmf") as infile:
        infile.read(mesh)
        infile.read(mvc, "subdomains")

    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_values(domains.array()+1)

    # Import mesh boundaries
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
    with XDMFFile(MPI.comm_world, "mesh/Stenosis_400K/mf.xdmf") as infile:
        infile.read(mvc, "boundaries")

    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    boundaries.array()[boundaries.array()> 1e10] = 0

    return mesh, domains, boundaries

class Noise(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, value, x):
        value[0] = np.random.normal(0, 0.001)
  
def create_bcs(DVP, boundaries, Re, mu_f, rho_f, D, **namespace):
    """
    Initiate the solution using boundary conditions as well as defining boundary conditions. 
    """
    average_inlet_velocity = get_ave_inlet_velocity(Re, mu_f, rho_f, D)
    default_variables.update(ave_inlet_velocity=average_inlet_velocity)
    inflow_prof = get_inflow_prof(average_inlet_velocity, D)
    
    noise = Noise()

    wallId = 1
    inletId = 3
    outletId = 2
    
    # generate functions of the initial solution from expressions
     # Define boundary conditions for the velocity 
    bc_u_inlet_x = DirichletBC(DVP.sub(1).sub(0), inflow_prof, boundaries, inletId)
    bc_u_inlet_y = DirichletBC(DVP.sub(1).sub(1), noise, boundaries, inletId)
    bc_u_inlet_z = DirichletBC(DVP.sub(1).sub(2), noise, boundaries, inletId)

    # boundary condition for the wall. No slip condition for the velocity.
    # This should be added at the end of the list of boundary conditions
    # to make sure that this is enforced.
    bc_u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, wallId)
    # Zero Dirichlet BC for pressure at the outlet
    bcp = DirichletBC(DVP.sub(2), 0, boundaries, outletId)

    bcs = [bc_u_inlet_x, bc_u_inlet_y, bc_u_inlet_z, bc_u_wall, bcp]
   
    return dict(bcs=bcs)


def get_ave_inlet_velocity(Re, mu_f, rho_f, D,**NS_namespace):
    average_inlet_velocity = Re*mu_f/D/rho_f
    return average_inlet_velocity


def get_inflow_prof(average_inlet_velocity, D, **NS_namespace):
    u_inflow = Expression('A*2*(1-((x[1]*x[1])+(x[2]*x[2]))*4/(D*D))', degree=2, A=average_inlet_velocity, D=D)
    return u_inflow