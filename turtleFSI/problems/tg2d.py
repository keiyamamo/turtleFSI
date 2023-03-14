import matplotlib.pyplot as plt
import pickle
from os import path
import sys
try:
    import pygmsh
except ImportError:
    pass

from dolfin import *
from turtleFSI.problems import *

"""
Taylor-Green vortex in 2D with fixed domain size
This problem can be used to test the accuracy of the fluid solver
"""

# Override some problem specific parameters
def set_problem_parameters(default_variables, **namespace):
    default_variables.update(dict(
        mu_f=0.01, # dynamic viscosity of fluid, 0.01 as kinematic viscosity
        T=1,
        dt=0.015625,
        theta=0.5, # Crank-Nicolson
        rho_f = 1, # density of fluid
        folder="tg2d_results",
        solid = "no_solid",
        extrapolation="no_extrapolation", # first try with static mesh
        plot_interval=100,
        save_step=10,
        checkpoint_step=100,
        compute_error=100,
        L = 2.,
        v_deg=2,
        p_deg=1,
        atol=1e-10,
        rtol=1e-10,
        total_error_v = 0,
        total_error_p = 0,
        mesh_size=0.25,
        mesh_type="unstructured",
        external_mesh=True
        ))

    return default_variables

def create2Dmesh(msh):
    # remove z coordinate
    msh.points = msh.points[:, :2]
    nodes = msh.points
    cells = msh.cells_dict["triangle"].astype(np.uintp)
    mesh = Mesh()
    editor = MeshEditor()
    # point, interval, triangle, quadrilateral, hexahedron
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(len(nodes))
    editor.init_cells(len(cells))
    [editor.add_vertex(i, n) for i, n in enumerate(nodes)]
    [editor.add_cell(i, n) for i, n in enumerate(cells)]
    editor.close()
    return mesh

def fac_mesh(mesh_size):
    with pygmsh.geo.Geometry() as geom:
        geom.add_rectangle(
            -1, 1, -1, 1, 0.0, mesh_size=mesh_size
        )
        mesh = geom.generate_mesh()
        mesh = create2Dmesh(mesh)
    return mesh

class Wall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

def get_mesh_domain_and_boundaries(mesh_size, mesh_type, external_mesh, **namespace):
    """
    Here, we create unstructured mesh using gmsh. 
    If pygmsh is not installed, we use default mesh from dolfin, which is structured.

    args:
        mesh_size: scaling factor for mesh size in gmsh. The lower the value, the finer the mesh.
    """
    if external_mesh:
        mesh = Mesh()
        with XDMFFile("mesh/UnitSquare/unitsquare.xdmf") as infile:
            infile.read(mesh)
        info_blue("Loaded external mesh")
    elif "pygmsh" in sys.modules and mesh_type == "unstructured":
        mesh = fac_mesh(mesh_size)
        # In case of MPI, redistribute the mesh to all processors
        MeshPartitioning.build_distributed_mesh(mesh)
    else:
        mesh = RectangleMesh(Point(-1, -1), Point(1, 1), 20, 20)
      
    # Mark the boundaries
    Allboundaries = DomainBoundary()
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    Allboundaries.mark(boundaries, 0) 
    wall = Wall()
    wall.mark(boundaries, 1)
    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_all(1)

    return mesh, domains, boundaries

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
 

def create_bcs(DVP, boundaries, **namespace):
    """
    Apply pure DirichletBC for deformation, velocity using analytical solution.
    """
    bcs = []
    velocity = analytical_velocity()
    p_bc_val = analytical_pressure()
    # Deformation is prescribed over the entire domain while the velocity is prescribed on the boundary
    u_bc = DirichletBC(DVP.sub(1), velocity, boundaries, 1)
    p_bc = DirichletBC(DVP.sub(2), p_bc_val, top_right_front_point, method="pointwise")    
    
    bcs.append(u_bc)
    bcs.append(p_bc)
    
    return dict(bcs=bcs,  velocity=velocity, p_bc_val=p_bc_val)
    
def initiate(dvp_, DVP, **namespace):
    """
    Initialize solution using analytical solution.
    """
    inital_velocity = analytical_velocity()
    inital_pressure = analytical_pressure()
    # generate functions of the initial solution from expressions
    ui = interpolate(inital_velocity, DVP.sub(1).collapse())
    pi = interpolate(inital_pressure, DVP.sub(2).collapse())
    # assign the initial solution to dvp_
    assign(dvp_["n"].sub(1), ui)
    assign(dvp_["n-1"].sub(1), ui)
    assign(dvp_["n"].sub(2), pi)
    assign(dvp_["n-1"].sub(2), pi)

    return dict(dvp_=dvp_)

def pre_solve(t, velocity, p_bc_val, **namespace):
    """
    update the boundary condition as boundary condition is time-dependent
    """ 
    velocity.t = t
    p_bc_val.t = t

    return dict(velocity=velocity, p_bc_val=p_bc_val)

def post_solve(DVP, dt, dvp_, total_error_v, total_error_p, velocity, p_bc_val, **namespace):
    """
    Compute errors after solving 
    """
    # Get deformation, velocity, and pressure
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True) 
    
    ve = interpolate(velocity, DVP.sub(1).collapse())
    pe = interpolate(p_bc_val, DVP.sub(2).collapse()) 
    E_v = errornorm(ve, v, norm_type="L2")
    E_p = errornorm(pe, p, norm_type="L2")

    total_error_v += E_v*dt
    total_error_p += E_p*dt

    if MPI.rank(MPI.comm_world) == 0:
        print("velocity error:", E_v)
        print("pressure error:", E_p)
  
    return dict(total_error_v=total_error_v, total_error_p=total_error_p)                 
      
def finished(total_error_v, total_error_p, mesh_size, dt, results_folder, **namespace):

    if MPI.rank(MPI.comm_world) == 0:
        print("total error for the velocity: ", total_error_v)
        print("total error for the pressure: ", total_error_p)

    save_data = dict(total_error_v=total_error_v, total_error_p=total_error_p, mesh_size=mesh_size, dt=dt)
    file_name = f'mesh_size_{mesh_size}_dt_{dt}.pickle'
    with open(path.join(results_folder, file_name), 'wb') as f:
        pickle.dump(save_data, f)