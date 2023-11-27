import numpy as np
import os
from dolfin import *
from turtleFSI.problems import *


def set_problem_parameters(default_variables, **namespace):
    # define and set problem variables values
    default_variables.update(dict(
        T=10,                               # Simulation end time
        dt=0.1,                            # Time step size
        theta=0.51,                         # Theta scheme (implicit/explicit time stepping): 0.5 + dt
        atol=1e-7,                           # Absolute tolerance in the Newton solver
        rtol=1e-7,                           # Relative tolerance in the Newton solver
        dx_f_id=1,                           # ID of marker in the fluid domain
        rho_f=1,                       # Fluid density [kg/m3]
        mu_f=1,                         # Fluid dynamic viscosity [Pa.s]
        save_solution_after_tstep=50,         # Save solution after this time step

        # Solid parameters
        solid = "no_solid",               # no solid
        extrapolation="no_extrapolation", # no extrapolation since the domain is fixed
        d_deg = 1,
        v_deg = 2,
        p_deg = 1,
        inletId = 1,
        outletId = 2,
        wallId = 3,

        u_avg = 0.5,
        R = 1,
        
        recompute=15,                        # Number of iterations before recompute Jacobian. 
        recompute_tstep=50,                  # Number of time steps before recompute Jacobian. 
        save_step=1,                         # Save frequency of files for visualisation
        folder="pipe_laminar",              # Folder where the results will be stored
        checkpoint_step=50,                  # checkpoint frequency
        save_deg=1,
        volume_mesh_path = "/Users/keiyamamoto/Documents/turtleFSI/turtleFSI/mesh/Pipe_laminar/mesh.xdmf",
        surface_mesh_path = "/Users/keiyamamoto/Documents/turtleFSI/turtleFSI/mesh/Pipe_laminar/mf.xdmf",
    ))

    return default_variables


def get_mesh_domain_and_boundaries(volume_mesh_path, surface_mesh_path, **namespace):
    # Import mesh file
    mesh = Mesh()
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with XDMFFile(MPI.comm_world, volume_mesh_path) as infile:
        infile.read(mesh)
        infile.read(mvc, "name_to_read")

    domains = MeshFunction("size_t", mesh, mesh.geometry().dim())
    domains.set_values(domains.array()+1)


    # Import mesh boundaries
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
    with XDMFFile(MPI.comm_world, surface_mesh_path) as infile:
        infile.read(mvc, "name_to_read")

    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)


    # # Print pvd files for domains and boundaries
    # ff = File("boundaries.pvd")
    # ff << boundaries 
    # ff = File("domains.pvd")
    # ff << domains

    # exit(1)

    return mesh, domains, boundaries


    
class InflowProfile(UserExpression):
    def __init__(self, R, u_avg, **kwargs):
        super().__init__(kwargs)
        self.R = R
        self.average_inlet_velocity = u_avg
        

    def eval(self, value, x):
        value[0] = self.average_inlet_velocity * 2 * (self.R*self.R-((x[1]*x[1])+(x[2]*x[2])) * 1 / (self.R*self.R))
        value[1] = 0
        value[2] = 0
    
    def value_shape(self):
        return (3,)
    

def initiate(mesh, v_deg, results_folder, **namespace):
    Vv = VectorFunctionSpace(mesh, "CG", v_deg)
    u_mean = Function(Vv)
    if MPI.rank(MPI.comm_world) == 0:
        os.makedirs(os.path.join(results_folder, "Solutions"))
    solution_path = os.path.join(results_folder, "Solutions")
    solution_mesh_path = os.path.join(solution_path, "mesh.h5")
    solution_velocity_path = os.path.join(solution_path, "u.h5")
    solution_pressure_path = os.path.join(solution_path, "p.h5")
    solution_u_mean_path = os.path.join(solution_path, "u_mean.h5")
    solution_files = {"solution_mesh" : solution_mesh_path, "solution_v" : solution_velocity_path, "solution_p" : solution_pressure_path, "solution_u_mean" : solution_u_mean_path}
    #  Save mesh as HDF5 file for post processing
    boundaries = MeshFunction("size_t", mesh, 2, mesh.domains())
    boundaries.set_values(boundaries.array()+1)
    with HDF5File(MPI.comm_world, solution_mesh_path, "w") as mesh_file:
        mesh_file.write(mesh, "mesh")
        mesh_file.write(boundaries, "boundaries")
    
    return dict(u_mean=u_mean, solution_files=solution_files)
  
def create_bcs(DVP, boundaries, u_avg, R, inletId, outletId, wallId, mesh, **namespace):
    """
    Initiate the solution using boundary conditions as well as defining boundary conditions. 
    """
    inflow_prof = InflowProfile(R=R, u_avg=u_avg, degree=2)
    
    # Define boundary conditions for the velocity 
    bc_u_inlet = DirichletBC(DVP.sub(1), inflow_prof, boundaries, inletId)
    bc_u_wall = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, wallId)

    # Zero Dirichlet BC for pressure at the outlet
    bcp = DirichletBC(DVP.sub(2), 0, boundaries, outletId)
    bcs = [bc_u_wall, bc_u_inlet, bcp]
    dsi = ds(inletId, domain=mesh, subdomain_data=boundaries)
    inlet_area = assemble(1*dsi)
    n = FacetNormal(mesh)
    if MPI.rank(MPI.comm_world) == 0:
        print("Inlet area: ", inlet_area)
    #NOTE: here it seems important to have inflow_prof as global variable, otherwise it will not work 
    return dict(bcs=bcs, inflow_prof=inflow_prof, dsi=dsi, inlet_area=inlet_area, n=n)


def post_solve(dvp_, u_mean, t, save_solution_after_tstep, solution_files, dt, inlet_area, n, dsi, **namespace):
    v = dvp_["n"].sub(1, deepcopy=True)
    flow_rate_inlet = abs(assemble(inner(v, n) * dsi))
    average_velocity_inlet = flow_rate_inlet / inlet_area

    if MPI.rank(MPI.comm_world) == 0:
        print("Flow rate at inlet: ", flow_rate_inlet)
        print("Average velocity at inlet: ", average_velocity_inlet)
    
    if t >= save_solution_after_tstep * dt :
        file_mode = "w" if not os.path.exists(solution_files["solution_v"]) else "a"

        # Extract solutions and assign to functions
        v = dvp_["n"].sub(1, deepcopy=True)
        p = dvp_["n"].sub(2, deepcopy=True)
        
        # Save velocity
        viz_u = HDF5File(MPI.comm_world, solution_files["solution_v"], file_mode=file_mode)
        viz_u.write(v, "/velocity", t /dt)
        viz_u.close()

        # Save pressure
        viz_p = HDF5File(MPI.comm_world, solution_files["solution_p"], file_mode=file_mode)
        viz_p.write(p, "/pressure", t /dt)
        viz_p.close()

        # Start averaging velocity w
        # Here, we accumulate the velocity filed in u_mean
        u_mean.vector().axpy(1, v.vector())

        return dict(u_mean=u_mean)    
    else:
        return None


def finished(u_mean, solution_files, save_solution_after_tstep, T, dt, **namespace):
    # Divide the accumulated velocity field by the number of time steps
    u_mean.vector()[:] = u_mean.vector()[:] / (T/dt - save_solution_after_tstep + 1)

    # Save u_mean as a XDMF file using the checkpoint
    u_mean.rename("u_mean", "u_mean")
    # Save u_mean
    with HDF5File(MPI.comm_world,  solution_files["solution_u_mean"], "w") as u_mean_file:
        u_mean_file.write(u_mean, "u_mean")