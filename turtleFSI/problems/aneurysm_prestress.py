import numpy as np

from turtleFSI.problems import *
from dolfin import *

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    x_sphere = 0.123043
    y_sphere = 0.13458
    z_sphere = 0.064187
    r_sphere = 0.004

    E_s_val = 1E6
    nu_s_val = 0.45
    mu_s_val = E_s_val/(2*(1+nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val*2.*mu_s_val/(1. - 2.*nu_s_val)

    default_variables.update(dict(
        # Time variables
        T=1.0, # Simulation end time
        dt=0.001, # Timne step size
        # Solver parameters
        atol=1e-5, # Absolute tolerance in the Newton solver
        rtol=1e-5,# Relative tolerance in the Newton solver
        recompute=15, # Recompute the Jacobian matix within time steps                                                                                    
        recompute_tstep=100, # Recompute the Jacobian matix over time steps (dangerous!)  
        theta=0.501, # Theta scheme (implicit/explicit time stepping)
        linear_solver="mumps",  # use list_linear_solvers() to check alternatives
        # boundary conditions
        mesh_file="file_case3",
        inlet_id=2,  # inlet
        inlet_outlet_s_id=11,  # also the "rigid wall" id for the stucture problem
        fsi_id=22,  # fsi surface
        rigid_id=11,  # "rigid wall" id for the fluid and mesh problem
        outer_id=33,  # outer surface
        dx_f_id=1,      # ID of marker in the fluid domain
        dx_s_id=2,      # ID of marker in the solid domain
        fsi_region=[x_sphere,y_sphere,z_sphere,r_sphere], # X, Y, and Z coordinate of FSI region center, radius of spherical deformable region (outside this region the walls are rigid)
        # Fluid parameters
        rho_f=1.000E3,    # Fluid density [kg/m3]
        mu_f=3.5E-3,       # Fluid dynamic viscosity [Pa.s]
        u_max = 0.74, # Maximum velocity of the inlet wheres the average velocity is 0.37
        # Pressure parameters
        p_val = 10932.404, # 82mmHg in Pa
        p_t_ramp_start = 0.2,
        p_t_ramp_end = 0.4,
        # Solid parameters
        rho_s=1.0E3,    # Solid density [kg/m3]
        mu_s=mu_s_val,     # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,      # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid Young's modulus [Pa]
        # Mesh lifting parameters
        extrapolation="laplace",  # laplace, elastic, biharmonic, no-extrapolation
        extrapolation_sub_type="constant",  # ["constant", "small_constant", "volume", "volume_change", "bc1", "bc2"]
        # I/O parameters
        save_step=50, # Save frequency of files for visualisation
        save_deg=1,          # Degree of the functions saved for visualisation '1' '2' '3' etc... (high value can slow down simulation significantly!)
        checkpoint_step=100, # 
        compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)
        folder="file_case3",
    ))

    return default_variables


def get_mesh_domain_and_boundaries(mesh_file, fsi_id, rigid_id, outer_id, fsi_region, **namespace):
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
    for submesh_facet in facets(mesh):
        idx_facet = boundaries.array()[i]
        if idx_facet == fsi_id:
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id  # changed "fsi" idx to "rigid wall" idx
        if idx_facet == outer_id:
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id  # changed "fsi" idx to "rigid wall" idx
        i += 1

    # # Checking boundaries and domains
    # f = File('toto.pvd')
    # f << boundaries
    # f << domains

    return mesh, domains, boundaries


class VelInPara(UserExpression):
    def __init__(self, t, t_ramp, u_max, n, dsi, mesh, **kwargs):
        self.t = t
        self.t_ramp = t_ramp
        self.u_max = u_max
        self.n = n
        self.dsi = dsi
        self.d = mesh.geometry().dim()
        self.x = SpatialCoordinate(mesh)
        # Compute area of boundary tesselation by integrating 1.0 over all facets
        self.A = assemble(Constant(1.0, name="one")*self.dsi)
        # Compute barycenter by integrating x components over all facets
        self.c = [assemble(self.x[i]*self.dsi) / self.A for i in range(self.d)]
        # Compute radius by taking max radius of boundary points
        self.r = np.sqrt(self.A / np.pi)
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        
    def eval(self, value, x):
        if self.t < self.t_ramp:
            interp_PA = self.u_max*(-0.5*np.cos((pi/(self.t_ramp))*(self.t)) + 0.5)  # Velocity initialisation with sigmoid
        else:
            interp_PA = self.u_max

        r2 = (x[0]-self.c[0])**2 + (x[1]-self.c[1])**2 + (x[2]-self.c[2])**2  # radius**2
        fact_r = 1 - (r2/self.r**2)

        value[0] = -self.n[0] * (interp_PA) *fact_r  # *self.t
        value[1] = -self.n[1] * (interp_PA) *fact_r  # *self.t
        value[2] = -self.n[2] * (interp_PA) *fact_r  # *self.t

    def value_shape(self):
        return (3,)


# Define the pressure ramp
class InnerP(UserExpression):
    def __init__(self, t, p_val, p_t_ramp_start, p_t_ramp_end, **kwargs):
        self.t = t 
        self.p_val = p_val
        self.p_t_ramp_start = p_t_ramp_start
        self.p_t_ramp_end = p_t_ramp_end
        super().__init__(**kwargs)

    def eval(self, value, x):
        if self.t < self.p_t_ramp_start:
            value[0] = 0.0
        elif self.t < self.p_t_ramp_end:
            value[0] = self.p_val*(-0.5*np.cos((pi/(self.p_t_ramp_end - self.p_t_ramp_start))*(self.t - self.p_t_ramp_start)) + 0.5) # Pressure initialisation with sigmoid
        else:
            value[0] = self.p_val

    def value_shape(self):
        return ()


def create_bcs(dvp_, DVP, mesh, boundaries, p_val, p_t_ramp_start, p_t_ramp_end, p_deg,
               fsi_id, inlet_id, inlet_outlet_s_id,
               rigid_id, psi, F_solid_linear, u_max, **namespace):

    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    # Define the pressure condition necessary to create the variational form (Neumann BCs)

    p_out_bc_val = InnerP(t=0.0, p_val=p_val, p_t_ramp_start=p_t_ramp_start, p_t_ramp_end=p_t_ramp_end, degree=p_deg)

    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    F_solid_linear += p_out_bc_val * inner(n('+'), psi('+'))*dSS(fsi_id)  # defined on the reference domain

    # Fluid velocity BCs
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni = np.array([assemble(n[i]*dsi) for i in range(ndim)])
    n_len = np.sqrt(sum([ni[i]**2 for i in range(ndim)]))  # Should always be 1!?
    normal = ni/n_len

    # new Parabolic profile
    u_inflow_exp = VelInPara(t=0.0, t_ramp=0.2, u_max=u_max, n=normal, dsi=dsi, mesh=mesh, degree=2)
    u_inlet = DirichletBC(DVP.sub(1), u_inflow_exp, boundaries, inlet_id)
    u_inlet_s = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)

    # Solid Displacement BCs
    d_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_id)
    d_inlet_s = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)
    d_rigid = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, rigid_id)

    # Assemble boundary conditions
    bcs = [u_inlet, d_inlet,  u_inlet_s, d_inlet_s, d_rigid]

    return dict(bcs=bcs, u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val,
                F_solid_linear=F_solid_linear)


def pre_solve(t, u_inflow_exp, p_out_bc_val, **namespace):
    # Update the time variable used for the inlet boundary condition
    u_inflow_exp.update(t)
    p_out_bc_val.t = t
    return dict(u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val)