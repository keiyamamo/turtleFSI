from turtleFSI.problems import *
from dolfin import *

"""
Note: Pressure is far from correct, but the velocity beahves nicely. Pressure gets wrong inside newton loop.
Date: 2022-10-10
"""

# Override some problem specific parameters
def set_problem_parameters(default_variables, **namespace):
    default_variables.update(dict(
        mu_f=0.01, # dynamic viscosity of fluid, 0.01 as kinematic viscosity
        T=1,
        dt=0.001,
        rho_f = 1,
        Nx=20, Ny=20,
        folder="tg2d_results",
        solid = "no_solid",
        extrapolation="no_extrapolation", # first try with static mesh
        plot_interval=100,
        save_step=10,
        checkpoint_step=100,
        compute_error=100,
        L = 2.,
        T_G = 4,
        A_value = 0.08,
        d_deg=2,
        v_deg=2,
        p_deg=1,
        total_error_v = 0,
        total_error_p = 0
        ))

    return default_variables

class Wall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

def get_mesh_domain_and_boundaries(L, Nx, Ny, **namespace):
    mesh = RectangleMesh(Point(-L / 2, -L / 2), Point(L / 2, L / 2), Nx, Ny)
    # Mark the boundaries
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    Allboundaries.mark(boundaries, 0) 
    wall = Wall()
    wall.mark(boundaries, 1)
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries

class analytical_displacement(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 0
        self.nu = 0.01
        self.L = 1.
        self.A = 0.08
        self.T_G = 4

    def eval(self, value, x):
        value[0] = self.A*sin(2*pi*self.t/self.T_G)*sin(2*pi*(x[1]+self.L/2)/self.L)
        value[1] = self.A*sin(2*pi*self.t/self.T_G)*sin(2*pi*(x[0]+self.L/2)/self.L)

    def value_shape(self):
        return (2,)

class analytical_velocity(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 0
        self.nu = 0.01
    
    def eval(self, value, x):
        value[0] = -sin(pi*x[1])*cos(pi*x[0])*exp(-2.*pi*pi*self.nu*self.t)
        value[1] = sin(pi*x[0])*cos(pi*x[1])*exp(-2.*pi*pi*self.nu*self.t)

    def value_shape(self):
        return (2,)

class analytical_pressure(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 0
        self.nu = 0.01

    def eval(self, value, x):
        value[0] = -(cos(2*pi*x[0])+cos(2*pi*x[1]))*exp(-4.*pi*pi*self.nu*self.t)/4.
    
    def value_shape(self):
        return ()

def top_right_front_point(x, on_boundary):
    tol = DOLFIN_EPS
    return near(x[0], 1.0, tol) and near(x[1], 1.0, tol)
 

def create_bcs(DVP, mesh, boundaries, psi, F_fluid_nonlinear, **namespace):
    """
    Apply pure DirichletBC for deformation, velocity using analytical solution.
    """
    bcs = []
    displacement = analytical_displacement()
    velocity = analytical_velocity()
    p_bc_val = analytical_pressure()
    # Deformation is prescribed over the entire domain while the velocity is prescribed on the boundary
    # d_bc = DirichletBC(DVP.sub(0), displacement, boundaries, 0)
    d_bc = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 1)
    u_bc = DirichletBC(DVP.sub(1), velocity, boundaries, 1)
    p_bc = DirichletBC(DVP.sub(2), p_bc_val, top_right_front_point, method="pointwise")    
    
    bcs.append(d_bc)
    bcs.append(u_bc)
    bcs.append(p_bc)
    
    return dict(bcs=bcs,  velocity=velocity, displacement=displacement, p_bc_val=p_bc_val)
    
def initiate(dvp_, DVP, **namespace):
    """
    Initialize solution using analytical solution.
    """
    inital_deformation = analytical_displacement()
    inital_velocity = analytical_velocity()
    inital_pressure = analytical_pressure()
    # generate functions of the initial solution from expressions
    # di = interpolate(inital_deformation, DVP.sub(0).collapse())
    ui = interpolate(inital_velocity, DVP.sub(1).collapse())
    pi = interpolate(inital_pressure, DVP.sub(2).collapse())
    # assign the initial solution to dvp_
    # assign(dvp_["n"].sub(0), di)
    # assign(dvp_["n-1"].sub(0), di)
    # assign(dvp_["n"].sub(1), ui)
    assign(dvp_["n-1"].sub(1), ui)
    # assign(dvp_["n"].sub(2), pi)
    assign(dvp_["n-1"].sub(2), pi)

    return dict(dvp_=dvp_)

def pre_solve(t, velocity, displacement, p_bc_val, **namespace):
    """
    update the boundary condition as boundary condition is time-dependent
    NOTE: it seems to work fine for updating the boundary condition
    """ 
    velocity.t = t
    # displacement.t = t
    p_bc_val.t = t

    # return dict(velocity=velocity, displacement=displacement, p_bc_val=p_bc_val, dvpp_=dvp_)
    return dict(velocity=velocity, p_bc_val=p_bc_val)

def post_solve(DVP, dt, dvp_, total_error_v, total_error_p, displacement, velocity, p_bc_val, **namespace):
    """
    Compute errors after solving 
    """
    # Get deformation, velocity, and pressure
    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True) 
    
    # de = interpolate(displacement, DVP.sub(1).collapse())
    ve = interpolate(velocity, DVP.sub(1).collapse())
    pe = interpolate(p_bc_val, DVP.sub(2).collapse()) 

    # compute error for the deformation
    # den = norm(de.vector())
    # de.vector().axpy(-1, d.vector())
    # error_d = norm(de.vector()) / den
    
    # compute error for the velocity 
    ven = norm(ve.vector())
    ve.vector().axpy(-1, v.vector())
    error_v = norm (ve.vector()) / ven
    
    total_error_v += error_v*dt
    # compute error for the pressure
    pen = norm(pe.vector())
    pe.vector().axpy(-1, p.vector())
    error_p = norm (pe.vector()) / pen
    
    total_error_p += error_p*dt

    if MPI.rank(MPI.comm_world) == 0:
        # print("deformation error:", error_d)
        print("velocity error:", error_v)
        print("pressure error:", error_p)
  
    return dict(total_error_v=total_error_v, total_error_p=total_error_p)                 
      
def finished(total_error_v, total_error_p, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        print("total error for the velocity: ", total_error_v)
        print("total error for the pressure: ", total_error_p)