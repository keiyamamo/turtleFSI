from os import path
import numpy as np
from numpy import genfromtxt

from dolfin import *
from turtleFSI.problems import *
from utils.Womersley import make_womersley_bcs, compute_boundary_geometry_acrn

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):
    
    # Read problem-specific variables from config file
    x_sphere = 0.123043
    y_sphere = 0.13458
    z_sphere = 0.064187
    r_sphere = 0.004
    dt = 0.00033964286
    mesh_path = "file_case9_el047"
    save_deg_sim = 2
    q_mean = 1.9275E-06

    # Overwrite default values
    E_s_val = 1E6
    nu_s_val = 0.45
    mu_s_val = E_s_val/(2*(1+nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val*2.*mu_s_val/(1. - 2.*nu_s_val)

    default_variables.update(dict(
        T=0.3,     # Simulation end time
        dt=dt,#0.00033964286, # Timne step size
        atol=1e-6, # Absolute tolerance in the Newton solver
        rtol=1e-6,# Relative tolerance in the Newton solver
        inlet_id=2,  # inlet
        inlet_outlet_s_id=11,  # also the "rigid wall" id for the stucture problem
        recompute=5, # Recompute the Jacobian matix within time steps                                                                                    
        recompute_tstep=2, # Recompute the Jacobian matix over time steps (dangerous!)  
        fsi_id=22,  # fsi surface
        rigid_id=11,  # "rigid wall" id for the fluid and mesh problem
        ds_s_id=[33],                     # ID of solid external wall (where we want to test Robin BC)
        outer_id=33,  # outer surface
        folder=mesh_path,
        mesh_file=mesh_path,
        q_file="MCA_10", # This is the location of CFD results used to prescribe the inlet velocity profile
        q_mean=q_mean,#1.9275E-06, # Problem specific
        theta=0.501, # Theta scheme (implicit/explicit time stepping)
        rho_f=1.000E3,    # Fluid density [kg/m3]
        mu_f=3.5E-3,       # Fluid dynamic viscosity [Pa.s]
        rho_s=1.0E3,    # Solid density [kg/m3]
        mu_s=mu_s_val,     # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,      # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid Young's modulus [Pa]
        dx_f_id=1,      # ID of marker in the fluid domain
        dx_s_id=2,      # ID of marker in the solid domain
        extrapolation="laplace",  # laplace, elastic, biharmonic, no-extrapolation
        extrapolation_sub_type="constant",  # ["constant", "small_constant", "volume", "volume_change", "bc1", "bc2"]
        compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)
        linear_solver="mumps",  # use list_linear_solvers() to check alternatives
        checkpoint_step=50, # CHANGE
        save_step=1, # Save frequency of files for visualisation
        save_deg=save_deg_sim,          # Degree of the functions saved for visualisation '1' '2' '3' etc... (high value can slow down simulation significantly!)
        fsi_region=[x_sphere,y_sphere,z_sphere,r_sphere], # X, Y, and Z coordinate of FSI region center, radius of spherical deformable region (outside this region the walls are rigid)
        p_wave_file = "p_t.csv", # File containing the pressure wave
        t_ramp=0.2,
        compute_acceleration=False,
    ))

    return default_variables


def get_mesh_domain_and_boundaries(mesh_file,fsi_region, fsi_id, rigid_id, outer_id, folder, **namespace):
    # Read mesh
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), "mesh/" + mesh_file + ".h5", "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    # Only considere FSI in domain within this sphere BC1
    sph_x = float(fsi_region[0])
    sph_y = float(fsi_region[1])
    sph_z = float(fsi_region[2])
    sph_rad = float(fsi_region[3])

    i = 0
    # Fluid-solid interface is by defualt assumed to have fsi id. Therefore, we compute the distance from the midpoint of the cell
    # to the center of the sphere and then change id from fsi to rigid wall if the distance is larger than the radius of the sphere
    for submesh_facet in facets(mesh):
        idx_facet = boundaries.array()[i]
        if idx_facet == fsi_id or idx_facet == outer_id:
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id  # changed "fsi" idx to "rigid wall" idx
        i += 1

    # f = File(f'{mesh_file}.pvd')
    # f << boundaries
    # f << domains
    # exit(1)

    return mesh, domains, boundaries


class InnerP(UserExpression):
    def __init__(self, t, t_ramp, t_p, p_PA, **kwargs):
        self.t = t
        self.t_ramp = t_ramp
        self.t_p = t_p
        self.p_PA = p_PA    
        self.p_0 = 0.0 # Initial pressure
        self.P = self.p_0 # Apply initial pressure to inner pressure variable
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        # apply a sigmoid ramp to the pressure 
        if self.t < self.t_ramp:
            ramp_factor = (-1/2)*np.cos(3.14159*self.t/self.t_ramp) + 1/2
        else:
            ramp_factor = 1.0
        if MPI.rank(MPI.comm_world) == 0:
            print("ramp_factor = {} m^3/s".format(ramp_factor))
        # Caclulate P as resistance boundary condition multiplied by ramp factor
        self.P = ramp_factor * (np.interp(self.t, self.t_p, self.p_PA))
        if MPI.rank(MPI.comm_world) == 0:
            print("P = {} Pa".format(self.P))

    def eval(self, value, x):
        value[0] = self.P 

    def value_shape(self):
        return ()


def create_bcs(t, DVP, mesh, boundaries, mu_f, t_ramp,
               fsi_id, inlet_id, inlet_outlet_s_id,
               rigid_id, psi, F_solid_linear, p_deg, q_file, q_mean, p_wave_file, **namespace):

    # Load normalized time and flow rate values
    t_values, Q_ = np.loadtxt(path.join(path.dirname(path.abspath(__file__)), q_file)).T
    Q_values = q_mean * Q_  # Specific flow rate = Normalized flow wave form * Prescribed flow rate
    tmp_area, tmp_center, tmp_radius, tmp_normal = compute_boundary_geometry_acrn(mesh, inlet_id, boundaries)

    # Create Womersley boundary condition at inlet
    inlet = make_womersley_bcs(t_values, Q_values, mesh, mu_f, tmp_area, tmp_center, tmp_radius, tmp_normal, DVP.sub(1).sub(0).ufl_element())

    # Initialize inlet expressions with initial time
    for uc in inlet:
        uc.set_t(t)

    # Create Boundary conditions for the velocity
    u_inlet = [DirichletBC(DVP.sub(1).sub(i), inlet[i], boundaries, inlet_id) for i in range(3)]
    u_inlet_s = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)

    # Solid Displacement BCs
    d_inlet = DirichletBC(DVP.sub(0), (0.0, 0.0, 0.0), boundaries, inlet_id)
    d_inlet_s = DirichletBC(DVP.sub(0), (0.0, 0.0, 0.0), boundaries, inlet_outlet_s_id)
    d_rigid = DirichletBC(DVP.sub(0), (0.0, 0.0, 0.0), boundaries, rigid_id)

    # Assemble boundary conditions
    bcs = u_inlet + [d_inlet, u_inlet_s, d_inlet_s, d_rigid]

    # Define the pressure condition (apply to inner surface, numerical instability results from applying to outlet)
    #dso = ds(outlet_id1, domain=mesh, subdomain_data=boundaries) # Outlet surface # Maybe try applying to all outlets???
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    p_t_file = genfromtxt(p_wave_file, delimiter=',')
    t_pressure=p_t_file[1:,0]
    pressure_PA=p_t_file[1:,1]
    p_out_bc_val = InnerP(t=0.0, t_p=t_pressure, t_ramp=t_ramp, p_PA=pressure_PA, degree=p_deg)
    n = FacetNormal(mesh)
    F_solid_linear += p_out_bc_val * inner(n('+'), psi('+'))*dSS(fsi_id)  # defined on the reference domain

    return dict(bcs=bcs, inlet=inlet, p_out_bc_val=p_out_bc_val, F_solid_linear=F_solid_linear)


def initiate(DVP, visualization_folder, compute_acceleration, **namespace):
    if compute_acceleration:
    # create a function for storing acceleration of the solid
        a = Function(DVP.sub(0).collapse())
        v_n = Function(DVP.sub(1).collapse())
        v_nm1 = Function(DVP.sub(1).collapse())
        a_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("acceleration.xdmf")))
        a_file.parameters["flush_output"] = True
        a_file.parameters["rewrite_function_mesh"] = False
        return dict(a=a, a_file=a_file, v_n=v_n, v_nm1=v_nm1)
    else:
        return dict(a=None, a_file=None, v_n=None, v_nm1=None)


def pre_solve(t, inlet, p_out_bc_val, **namespace):
    for uc in inlet:
        # Update the time variable used for the inlet boundary condition
        uc.set_t(t)

        # Multiply by cosine function to ramp up smoothly over time interval 0-250 ms
        if t < 0.25:
            uc.scale_value = -0.5 * np.cos(np.pi * t / 0.25) + 0.5
        else:
            uc.scale_value = 1.0

    # Update pressure condition
    p_out_bc_val.update(t)

    return dict(inlet=inlet, p_out_bc_val=p_out_bc_val)

def post_solve(a, dt, dvp_, t, a_file, v_n, v_nm1, compute_acceleration, **namespace):
    # Compute acceleration of the solid
    """
    There are two ways to compute the acceleration of the solid:
    1. Compute the acceleration from the displacement
    2. Compute the acceleration from the velocity
    However, the acceleration computed from the displacement is better than the acceleration computed from the velocity
    because it will highlight the accelerartion of the solid.
    """
    if compute_acceleration:
        d_n = dvp_["n"].sub(0, deepcopy=True)
        d_nm1 = dvp_["n-1"].sub(0, deepcopy=True)
        d_nm2 = dvp_["n-2"].sub(0, deepcopy=True)

        # Compute the velocity from the displacement
        v_n.vector()[:] = (d_n.vector()[:] - d_nm1.vector()[:])/dt
        v_nm1.vector()[:] = (d_nm1.vector()[:] - d_nm2.vector()[:])/dt

        a.vector()[:] = (v_n.vector()[:] - v_nm1.vector()[:])/dt

        # write acceleration to file
        a.rename("Acceleration", "a")
        a_file.write(a, t)
        # initialize the acceleration for the next time step
        a.vector().zero()
        v_n.vector().zero()
        v_nm1.vector().zero()



def finished(**namespace):
    with open("finished", mode='a'): pass



