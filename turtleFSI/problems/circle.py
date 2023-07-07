import numpy as np
from os import path
from dolfin import *
from turtleFSI.problems import *


# The "ghost_mode" has to do with the assembly of form containing the facet normals n('+') within interior boundaries (dS). For 3D mesh the value should be "shared_vertex", for 2D mesh "shared_facet", the default value is "none".
parameters["ghost_mode"] = "shared_facet" #2D mesh case
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):
    # set problem parameters values
    E_s_val = 1E6                              # Young modulus (elasticity) [Pa] Increased a lot for the 2D case
    nu_s_val = 0.45                            # Poisson ratio (compressibility)
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # Shear modulus
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    # define and set problem variables values
    default_variables.update(dict(
        T=0.5,                               # Simulation end time
        dt=0.0005,                           # Time step size
        theta=0.501,                         # Theta scheme (implicit/explicit time stepping): 0.5 + dt
        atol=1e-7,                           # Absolute tolerance in the Newton solver
        rtol=1e-7,                           # Relative tolerance in the Newton solver
        robin_bc=True,                       # Robin boundary condition
        dx_s_id=1,                           # ID of marker in the solid domain
        ds_s_id=[1],                         # IDs of solid external boundaries for Robin BC (external wall + solid outlet)
        rho_f=1.025E3,                       # Fluid density [kg/m3]
        mu_f=1.0,                            # Fluid dynamic viscosity [Pa.s]
        rho_s=2E3,                           # Solid density [kg/m3]
        mu_s=mu_s_val,                       # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,                       # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,               # Solid 1rst Lamé coef. [Pa]
        k_s = 51E5,                          # elastic response necesary for RobinBC
        c_s = 0.1,                             # viscoelastic response necesary for RobinBC
        extrapolation="no_extrapolation",             # laplace, elastic, biharmonic, no-extrapolation
        recompute=5,                         # Number of iterations before recompute Jacobian. 
        recompute_tstep=10,                  # Number of time steps before recompute Jacobian. 
        save_step=1,                         # Save frequency of files for visualisation
        folder="circle",                     # Folder where the results will be stored
        checkpoint_step=50,                  # checkpoint frequency
        fluid="no_fluid",                    # Do not solve for the fluid
        # gravity = 2.0,
        save_deg=1                           
    ))

    return default_variables


def get_mesh_domain_and_boundaries(folder, **namespace):
    # Import mesh file
    mesh = Mesh()
    with XDMFFile("mesh/Circle/mesh.xdmf") as infile:
        infile.read(mesh)
    # # Rescale the mesh coordinated from [mm] to [m]
    x = mesh.coordinates()
    scaling_factor = 0.1  # from mm to m
    x[:, :] *= scaling_factor
    mesh.bounding_box_tree().build(mesh)

    # Define mesh domains
    domains = MeshFunction("size_t", mesh, mesh.topology().dim()) 
    with XDMFFile("mesh/Circle/mesh.xdmf") as infile:
        infile.read(domains, "name_to_read")

    # Import mesh boundaries
    boundaries_mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1) 
    with XDMFFile("mesh/Circle/facet_mesh.xdmf") as infile:
        infile.read(boundaries_mvc, "name_to_read")

    boundaries = cpp.mesh.MeshFunctionSizet(mesh, boundaries_mvc)
   
    info_blue("Obtained mesh, domains and boundaries.")

    # Print pvd files for domains and boundaries
    # ff = File("boundaries.pvd")
    # ff << boundaries 
    # exit(1)
    # ff = File("domains.pvd")
    # ff << domains
    # exit(1)

    return mesh, domains, boundaries


class BodyForceImpulse(UserExpression):
    def __init__(self, force_val, t_start,t_end,t,**kwargs):
        self.force_val = force_val
        self.t_start = t_start
        self.t_end = t_end
        self.force = 0.0
        self.t = t

        super().__init__(**kwargs)

    def update(self, t):
        self.t = t

    def eval(self, value,x):
        if self.t >= self.t_start and self.t<=self.t_end:
            self.force = self.force_val 
        else:
            self.force = 0.0
        # Impluse should be applied in the y direction
        value[1] = self.force

    def value_shape(self):
        return (2,)


# Create boundary conditions
def create_bcs(DVP, boundaries, dx_s, psi, F_solid_linear, **namespace):

    bcs=[]

    impulse_force = BodyForceImpulse(force_val=1e5, t_start=0.005,t_end=0.008,t=0.0)
    F_solid_linear -= inner(impulse_force, psi)*dx_s[0]

    return dict(bcs=bcs, F_solid_linear=F_solid_linear, impulse_force=impulse_force)

# def initiate(dvp_,DVP, **namespace):

#     center_point = np.array([0.2, 1.3])
#     d_probe = Probe(center_point, DVP.sub(0))
#     d_probe(dvp_["n"].sub(0, deepcopy=True))

#     return dict(d_probe=d_probe)


def pre_solve(t, impulse_force, **namespace):
    # Update the time variable used for the inlet boundary condition
    impulse_force.update(t)
    return dict(impulse_force=impulse_force)


# def post_solve(d_probe, dvp_,  **namespace):
#     d_probe(dvp_["n"].sub(0, deepcopy=True))

def finished(results_folder, default_variables, **namespace):
    with open(path.join(results_folder, 'params.txt'), 'w') as par:
        for key, value in default_variables.items(): 
            par.write('%s: %s\n' % (key, value))
    