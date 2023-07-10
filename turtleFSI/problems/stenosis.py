import numpy as np

from turtleFSI.problems import *
from dolfin import DirichletBC, cpp, MeshValueCollection, UserExpression, MeshFunction, MPI


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
        save_step = 1,
        folder="stenosis_results",
        recompute=10,
        recompute_tstep=10,
        save_deg=1,
        d_deg = 1,
        v_deg = 2,
        p_deg = 1,

        atol=1e-6, # Absolute tolerance in the Newton solver
        rtol=1e-6,# Relative tolerance in the Newton solver

        volume_mesh_path = "mesh/Stenosis_400K/mesh.xdmf",
        surface_mesh_path = "mesh/Stenosis_400K/mf.xdmf",
        ))

    return default_variables


def get_mesh_domain_and_boundaries(volume_mesh_path, surface_mesh_path, **namespace):
    # Import mesh file
    mesh = Mesh()
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with XDMFFile(MPI.comm_world, volume_mesh_path) as infile:
        infile.read(mesh)
        infile.read(mvc, "subdomains")

    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_values(domains.array()+1)

    # Import mesh boundaries
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
    with XDMFFile(MPI.comm_world, surface_mesh_path) as infile:
        infile.read(mvc, "boundaries")

    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    return mesh, domains, boundaries

    
class InflowProfile(UserExpression):
    def __init__(self, Re, mu_f, rho_f, D, **kwargs):
        super().__init__(kwargs)
        self.Re = Re
        self.mu_f = mu_f
        self.rho_f = rho_f
        self.D = D
        self.average_inlet_velocity = self.Re*self.mu_f/self.D/self.rho_f

    def eval(self, value, x):
        value[0] = self.average_inlet_velocity* 2 * (1-((x[1]*x[1])+(x[2]*x[2])) * 4 / (self.D*self.D))
        value[1] = np.random.normal(0, 0.001)
        value[2] = np.random.normal(0, 0.001)
    
    def value_shape(self):
        return (3,)

  
def create_bcs(DVP, boundaries, Re, mu_f, rho_f, D, **namespace):
    """
    Initiate the solution using boundary conditions as well as defining boundary conditions. 
    """
    inflow_prof = InflowProfile(Re=Re, mu_f=mu_f, rho_f=rho_f, D=D, degree=2)


    wallId = 1
    inletId = 3
    outletId = 2
    
    # Define boundary conditions for the velocity 
    bc_u_inlet = DirichletBC(DVP.sub(1), inflow_prof, boundaries, inletId)
    bc_u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, wallId)

    # Zero Dirichlet BC for pressure at the outlet
    bcp = DirichletBC(DVP.sub(2), 0, boundaries, outletId)

    bcs = [bc_u_wall, bc_u_inlet, bcp]
    #NOTE: here it seems important to have inflow_prof as global variable, otherwise it will not work 
    return dict(bcs=bcs, inflow_prof=inflow_prof)

