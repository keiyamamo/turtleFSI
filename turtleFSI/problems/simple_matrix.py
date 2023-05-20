from dolfin import *
from turtleFSI.problems import *

"""
Author: Kei Yamamoto <keiya@simula.no>
Last updated: 20/05/2023

This propblem is designed to understand the matrix structure of the FSI problem
"""

# Override some problem specific parameters
def set_problem_parameters(default_variables, **namespace):
    default_variables.update(dict(
        mu_f=1,                        # dynamic viscosity of fluid, 0.01 as kinematic viscosity
        T=10,
        dt=1,
        theta=0.5,                        # Crank-Nicolson
        rho_f = 1,                        # density of fluid
        rho_s = 1,                        # density of solid
        folder="matrix_results",
        fluid = "fluid",
        solid = "solid",
        extrapolation="laplace", 
        save_step=1,
        d_deg=1,
        v_deg=1,
        p_deg=1,
        N=1,                              # number of points along x or y axis when creating structured mesh
        dx_f_id=1,       # Domain id of the fluid domain
        dx_s_id=2,       # Domain id of the solid domain
        ))

    return default_variables

def get_mesh_domain_and_boundaries(N,**namespace):
    """
    Here, we create a structured mesh with two elements, one for fluid and one for solid
    Left upper trigle is fluid and right lower triangle is solid
    """
    info_blue("Creating structured mesh")
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), N, N, "right")
      
    # Mark the boundaries
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    Allboundaries.mark(boundaries, 0) 
    Left_upper = AutoSubDomain(lambda x: x[0] <= x[1])
    Right_lower = AutoSubDomain(lambda x: x[1] <= x[0])
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    Left_upper.mark(domains, 1)     # fluid domain, the marker must be the same as dx_f_id
    Right_lower.mark(domains, 2)    # solid domain, the marker must be the same as dx_s_id
    
    return mesh, domains, boundaries

class analytical_displacement(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def eval(self, value, x):
        value[0] = 0
        value[1] = 0

    def value_shape(self):
        return (2,)

class analytical_velocity(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def eval(self, value, x):
        value[0] = 1
        value[1] = 1

    def value_shape(self):
        return (2,)

class analytical_pressure(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, value, x):
        value[0] = 2
    
    def value_shape(self):
        return ()

def create_bcs(**namespace):
    """
    Return an empty bcs
    """
    bcs = []
    return dict(bcs=bcs)
       
def initiate(dvp_, DVP, **namespace):
    """
    Initialize solution using analytical solution.
    """
    iniial_displacement = analytical_displacement()
    inital_velocity = analytical_velocity()
    inital_pressure = analytical_pressure()
    # generate functions of the initial solution from expressions
    di = interpolate(iniial_displacement, DVP.sub(0).collapse())
    ui = interpolate(inital_velocity, DVP.sub(1).collapse())
    pi = interpolate(inital_pressure, DVP.sub(2).collapse())
    # assign the initial solution to dvp_
    assign(dvp_["n"].sub(0), di)
    assign(dvp_["n"].sub(1), ui)
    assign(dvp_["n"].sub(2), pi)

    return dict(dvp_=dvp_)
    

def pre_solve(**namespace):
    """
    Here maybe I can save the matrix and rhs for each time step.
    """ 
    from IPython import embed; embed(); exit(1)
    pass

def post_solve(**namespace):
    pass            
      
def finished(**namespace):
    pass