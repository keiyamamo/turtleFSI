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
        solid_vel=0.1, # this is the velocity of the wall with prescribed displacement
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
    sideY = AutoSubDomain(lambda x: (x[1] < negEnd or x[1] > posEnd))
    sideZ = AutoSubDomain(lambda x: (x[2] < negEnd or x[2] > posEnd))

    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    Lwall.mark(boundaries, 1)
    Rwall.mark(boundaries, 2)

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


class InnerP(UserExpression):
    def __init__(self, t, t_start, t_end, P_final, **kwargs):
        self.t = t
        self.t_start = t_start
        self.t_end = t_end
        self.P_final = P_final
        self.P = 0.0
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        # apply a sigmoid ramp to the pressure
        if self.t < self.t_start:
            ramp_factor = 0.0
        elif self.t < self.t_end and self.t > self.t_start:
            ramp_factor = -0.5 * np.cos(np.pi * (self.t - self.t_start) / (self.t_end - self.t_start)) + 0.5
        else:
            ramp_factor = 1.0
        self.P = ramp_factor * self.P_final

        if MPI.rank(MPI.comm_world) == 0:
            print("P = {} Pa".format(self.P))

    def eval(self, value, x):
        value[0] = self.P

    def value_shape(self):
        return ()
    

def top_left_point(x, on_boundary):
    """
    Fixing the point for avoiding rotation of the mesh.
    """
    tol = DOLFIN_EPS
    return near(x[0], 0.003, tol) and near(x[1], 0.001, tol) and near(x[2], 0.001, tol)


def bottom_right_point(x, on_boundary):
    """
    Fixing the point for avoiding rotation of the mesh.
    """
    tol = DOLFIN_EPS
    return near(x[0], 0.003, tol) and near(x[1], 0.0, tol) and near(x[2], 0.0, tol)


def top_right_point(x, on_boundary):
    tol = DOLFIN_EPS
    return near(x[0], 0.003, tol) and near(x[1], 0.001, tol) and near(x[2], 0.0, tol)



def create_bcs(F_solid_linear, DVP, boundaries, P_final, t_start_p, t_end_p, mesh, psi, **namespace):
    # Apply pressure at the fsi interface by modifying the variational form
    p_out_bc_val = InnerP(t=0.0, t_start=t_start_p, t_end=t_end_p, P_final=P_final, degree=2)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    # defined on the reference domain
    # NOTE: ('+') implicitly assumes that the solid domain has a higher domain ID than the fluid domain
    F_solid_linear += p_out_bc_val * inner(n, psi) * ds(1)
    # Clamp on the right hand side
    u_rwall = DirichletBC(DVP.sub(0).sub(0), ((0.0)), boundaries, 2)

    bc_point1 = DirichletBC(DVP.sub(0), ((0, 0, 0)), top_left_point, method="pointwise")    
    bc_point2 = DirichletBC(DVP.sub(0), ((0, 0, 0)), bottom_right_point, method="pointwise")
    bc_point3 = DirichletBC(DVP.sub(0), ((0, 0, 0)), top_right_point, method="pointwise")
    
    bcs = [u_rwall, bc_point1, bc_point2, bc_point3]

    return dict(bcs=bcs, p_out_bc_val=p_out_bc_val, 
                F_solid_linear=F_solid_linear)


def pre_solve(t, p_out_bc_val, **namespace):
    """Update boundary conditions"""
    p_out_bc_val.update(t)
