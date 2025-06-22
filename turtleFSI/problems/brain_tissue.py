# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import *
from turtleFSI.problems import *



# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. Doesnt affect the speed
parameters["reorder_dofs_serial"] = False

def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    E_s_val = 1E6  # Young modulus (Pa)
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    E_s_val_brain = 1E5  # Young modulus (Pa) 100k Pa
    nu_s_val_brain = 0.499 # Poisson ratio, nearly incompressible
    mu_s_val_brain = E_s_val_brain / (2 * (1 + nu_s_val_brain))  # 0.345E6
    lambda_s_val_brain = nu_s_val_brain * 2. * mu_s_val_brain / (1. - 2. * nu_s_val_brain)
     
    default_variables.update(dict(
        # Temporal variables
        T=1,          # End time [s]
        dt=0.01,       # Time step [s]
        checkpoint_step=1000, # Checkpoint frequency
        theta=0.51,     # Temporal scheme
        save_step=1,
        P_final=10000,  # Steady State pressure applied to wall
        t_start_p=0.0,  # Start time for pressure application
        t_end_p=0.9,    # End time for pressure application

        # Physical constants
        gravity=None,   # Gravitational force [m/s**2]

        # Problem specific
        dx_s_id=[0, 1],     # Id of the solid domain
        folder="brain_tissue",          # Folder to store the results
        fluid="no_fluid",                 # Do not solve for the fluid
        extrapolation="no_extrapolation",  # No displacement to extrapolate
        solid_vel=0.0003, # this is the velocity of the wall with prescribed displacement
        solid_properties=[{"dx_s_id": 0, "material_model": "StVenantKirchoff", "rho_s": 1.0E3, "mu_s": mu_s_val,
                           "lambda_s": lambda_s_val},
                          {"dx_s_id": 1, "material_model": "StVenantKirchoff", "rho_s": 1.0E3, "mu_s": mu_s_val_brain,
                           "lambda_s": lambda_s_val_brain}],
        ))   

    return default_variables


def get_mesh_domain_and_boundaries(dx_s_id, **namespace):
    
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(0.003, 0.001, 0.001), 20, 6, 6)

    tol = 1E-14
    # Mark boundaries
    Lwall = AutoSubDomain(lambda x: (x[0]< tol))
    Rwall = AutoSubDomain(lambda x: (x[0]> 0.003 - tol))
    sideY = AutoSubDomain(lambda x: (x[1] < tol))
    sideZ = AutoSubDomain(lambda x: (x[2] < tol))

    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    Lwall.mark(boundaries, 1)
    Rwall.mark(boundaries, 2)
    sideY.mark(boundaries, 3)
    sideZ.mark(boundaries, 4)

    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(0)
    x_min = 0.0003
    i = 0
    for cell in cells(mesh):
        idx_cell = domains.array()[i]
        if idx_cell ==  dx_s_id[0]:
            mid = cell.midpoint()
            if mid.x() > x_min:
                domains.array()[i] = dx_s_id[1]
        i += 1
    
    return mesh, domains, boundaries

    
class PrescribedDisp(UserExpression):
    def __init__(self, solid_vel, **kwargs):
        self.solid_vel = solid_vel
        self.factor = 0

        super().__init__(**kwargs)

    def update(self, t):
        self.factor = t * self.solid_vel
        print('displacement = ', self.factor)

    def eval(self, value,x):
        value[0] = self.factor


def create_bcs(DVP,d_deg,solid_vel, boundaries, **namespace):
    # Sliding contact on 3 sides
    u_lwallX = DirichletBC(DVP.sub(0).sub(0), ((0.0)), boundaries, 2)
    u_CornerY = DirichletBC(DVP.sub(0).sub(1), ((0.0)), boundaries, 3)
    u_CornerZ = DirichletBC(DVP.sub(0).sub(2), ((0.0)), boundaries, 4)

    # Displacement on the right hand side (unconstrained in Y and Z)
    d_t = PrescribedDisp(solid_vel,degree=d_deg)
    u_rwall = DirichletBC(DVP.sub(0).sub(0), d_t, boundaries, 1)

    bcs = [u_lwallX, u_rwall,u_CornerY,u_CornerZ]

    return dict(bcs=bcs,d_t=d_t)


def pre_solve(t, d_t, **namespace):
    """Update boundary conditions"""
    d_t.update(t)
