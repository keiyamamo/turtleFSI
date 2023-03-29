from turtleFSI.problems import *
from dolfin import *

import pickle
from os import path

"""
Note: Pressure is far from correct, but the velocity beahves nicely. Pressure gets wrong inside newton loop. (2022-10-10)
      Pressure is better now, but the deformation is still wrong. (2023-03-21)
      The unique point of this problem is that the domain is moving at the outflow boundary by DirichletBC.
      We might have to add some extra terms that only appeear for special boundary condition. (2023-03-23)
"""

# Override some problem specific parameters
def set_problem_parameters(default_variables, **namespace):
    default_variables.update(dict(
        mu_f=0.025,
        rho_f=1,
        T=1,
        dt=0.001,
        Nx=40, Ny=40,
        theta = 0.501,
        folder="movingvortex_results",
        solid = "no_solid",
        extrapolation="laplace", 
        plot_interval=100,
        save_step=10,
        checkpoint_step=100,
        compute_error=100,
        L = 1.,
        T_G = 4,
        A_value = 0.08,
        d_deg=2,
        v_deg=2,
        p_deg=1,
        total_error_d =0,
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
        self.nu = 0.025
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
        self.nu = 0.025
        self.L = 1.
        self.A = 0.08
        self.T_G = 4
    
    def eval(self, value, x):
        value[0] = -sin(2*pi*x[1])*exp(-4*pi*pi*self.nu*self.t)
        value[1] = sin(2*pi*x[0])*exp(-4*pi*pi*self.nu*self.t)

    def value_shape(self):
        return (2,)

class analytical_pressure(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 0
        self.nu = 0.025
        self.L = 1.
        self.A = 0.08
        self.T_G = 4

    def eval(self, value, x):
        value[0] = -cos(2*pi*x[0])*cos(2*pi*x[1])*exp(-8*pi*pi*self.nu*self.t)
    
    def value_shape(self):
        return ()

def top_right_front_point(x, on_boundary):
    """
    This is actually not a good way to enforce the boundary condition.
    This point could be not on the node of the mesh and the boundary condition wouldn't be enforced.
    For example, Nx=10 and Ny=10, the point (0.25, 0.5) is not on the node of the mesh.
    """
    tol = DOLFIN_EPS
    return (abs(x[0] - 0.5) < tol) and (abs(x[1] - 0.5) < tol)

def p_zero(x, on_boundary):
    """
    Since we only have Neumann boundary condition for pressure, 
    we need to enforce p=0 on the boundary to make the problem well-posed.
    """
    return near(x[0], 0.25) and near(x[1], 0.5) 

def outflow(x, on_boundary):
    tol = DOLFIN_EPS
    return on_boundary and ( ((x[0] > tol) and near(x[1], 0.5)) or ((x[1] < tol) and near(x[0], 0.5)) or ((x[0] < tol) and near(x[1], -0.5)) or ((x[1] > tol) and near(x[0], -0.5)))

def create_bcs(DVP, mesh, boundaries, psi, F_fluid_nonlinear, **namespace):
    """
    Apply pure DirichletBC for deformation, velocity using analytical solution.
    """
    bcs = []
    displacement = analytical_displacement()
    velocity = analytical_velocity()
    p_bc_val = analytical_pressure()
    
    d_bc = DirichletBC(DVP.sub(0), displacement, boundaries, 1)
    u_bc = DirichletBC(DVP.sub(1), velocity, boundaries, 1)
    p_bc = DirichletBC(DVP.sub(2), Constant(0), p_zero, method="pointwise")
    
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
    di = interpolate(inital_deformation, DVP.sub(0).collapse())
    ui = interpolate(inital_velocity, DVP.sub(1).collapse())
    pi = interpolate(inital_pressure, DVP.sub(2).collapse())
    # assign the initial solution to dvp_
    assign(dvp_["n"].sub(0), di)
    assign(dvp_["n-1"].sub(0), di)
    assign(dvp_["n"].sub(1), ui)
    assign(dvp_["n-1"].sub(1), ui)
    assign(dvp_["n"].sub(2), pi)
    assign(dvp_["n-1"].sub(2), pi)

    return dict(dvp_=dvp_)

def pre_solve(t, velocity, displacement, p_bc_val, dvp_, DVP, **namespace):
    """
    update the boundary condition as boundary condition is time-dependent
    NOTE: it seems to work fine for updating the boundary condition
    """ 
    velocity.t = t
    displacement.t = t
    p_bc_val.t = t

    return dict(velocity=velocity, displacement=displacement, p_bc_val=p_bc_val, dvpp_=dvp_)

def post_solve(DVP, dt, dvp_, t, total_error_d, total_error_v, total_error_p, displacement, velocity, p_bc_val, **namespace):
    """
    Compute errors after solving 
    """
    # Get deformation, velocity, and pressure
    # d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True) 
    
    # de = interpolate(displacement, DVP.sub(0).collapse())
    ve = interpolate(velocity, DVP.sub(1).collapse())
    pe = interpolate(p_bc_val, DVP.sub(2).collapse()) 
    
    # compute error for the deformation
    # error_d = errornorm(de, d, norm_type="L2")
    error_v = errornorm(ve, v, norm_type="L2")
    error_p = errornorm(pe, p, norm_type="L2")
    
    # total_error_d += error_d*dt
    total_error_v += error_v*dt
    total_error_p += error_p*dt

    if MPI.rank(MPI.comm_world) == 0:
        # print("deformation error:", error_d)
        print("velocity error:", error_v)
        print("pressure error:", error_p)
  
    # return dict(total_error_d=total_error_d, total_error_v=total_error_v, total_error_p=total_error_p)                 
    return dict(total_error_v=total_error_v, total_error_p=total_error_p)                 
      
# def finished(total_error_d, total_error_v, total_error_p, dt, results_folder, **namespace):
def finished(total_error_v, total_error_p, dt, results_folder, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        # print("total error for the deformation: ", total_error_d)
        print("total error for the velocity: ", total_error_v)
        print("total error for the pressure: ", total_error_p)
    
    save_data = dict(total_error_v=total_error_v, total_error_p=total_error_p, dt=dt)
    # save_data = dict(total_error_d=total_error_d, total_error_v=total_error_v, total_error_p=total_error_p, dt=dt)
    file_name = f'results_dt_{dt}.pickle'
    with open(path.join(results_folder, file_name), 'wb') as f:
        pickle.dump(save_data, f)