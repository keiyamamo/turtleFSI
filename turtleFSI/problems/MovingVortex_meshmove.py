from turtleFSI.problems import *
from dolfin import *

import pickle
from os import path
import matplotlib.pyplot as plt
"""
This is a problem file to test mesh movement in turtleFSI. No fluid or solid is involved.
Confirmed that velocity and pressure is zero for all the time. 
"""


# Override some problem specific parameters
def set_problem_parameters(default_variables, **namespace):
    default_variables.update(dict(
        mu_f=0.025,
        rho_f=1,
        T=1,
        dt=0.001,
        Nx=20, Ny=20,
        folder="movingvortex_meshmove_results",
        solid = "no_solid",
        fluid = "no_fluid",
        extrapolation="laplace",
        plot_interval=100,
        save_step=1,
        checkpoint_step=100,
        compute_error=100,
        L = 1.,
        T_G = 4,
        A_value = 0.08,
        d_deg=2,
        v_deg=2,
        p_deg=1,
        total_error_d =0,
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

def create_bcs(DVP, boundaries, **namespace):
    """
    Apply pure DirichletBC for deformation, velocity using analytical solution.
    """
    bcs = []
    displacement = analytical_displacement()
    d_bc = DirichletBC(DVP.sub(0), displacement, boundaries, 1)
    bcs.append(d_bc)
    
    return dict(bcs=bcs, displacement=displacement)
    
def initiate(dvp_, DVP, **namespace):
    """
    Initialize solution using analytical solution.
    """
    inital_deformation = analytical_displacement()
    # generate functions of the initial solution from expressions
    di = interpolate(inital_deformation, DVP.sub(0).collapse())
    # assign the initial solution to dvp_
    assign(dvp_["n"].sub(0), di)
    assign(dvp_["n-1"].sub(0), di)
    return dict(dvp_=dvp_)

def pre_solve(t, displacement, dvp_, **namespace):
    """
    update the boundary condition as boundary condition is time-dependent
    NOTE: it seems to work fine for updating the boundary condition
    """ 
    displacement.t = t
    return dict(displacement=displacement, dvpp_=dvp_)

def post_solve(DVP, dt, dvp_, t, total_error_d, displacement, **namespace):
    """
    Compute errors after solving 
    """
    # Get deformation, velocity, and pressure
    d = dvp_["n"].sub(0, deepcopy=True)
    de = interpolate(displacement, DVP.sub(0).collapse())
    from IPython import embed; embed(); exit(1)
    #plot(de)
    #plot(d)
    #plt.show()
    # compute error for the deformation
    error_d = errornorm(de, d, norm_type="L2")
    total_error_d += error_d*dt
    if MPI.rank(MPI.comm_world) == 0:
        print("deformation error:", error_d)


  
    return dict(total_error_d=total_error_d)
      
def finished(total_error_d,dt, results_folder, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        print("total error for the deformation: ", total_error_d)
    
    save_data = dict(total_error_d=total_error_d)
    file_name = f'results_dt_{dt}.pickle'
    with open(path.join(results_folder, file_name), 'wb') as f:
        pickle.dump(save_data, f)