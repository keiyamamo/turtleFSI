from dolfin import *

from turtleFSI.problems import *

"""
Last update: 2023--6-19
2D tube case intended to simulate propagation of a pressure wave in a tube
"""

def set_problem_parameters(default_variables, **namespace):

    # Overwrite default values
    E_s_val = 1E6    # Young modulus (elasticity) [dyn/cm2]
    nu_s_val = 0.45   # Poisson ratio (compressibility)
    mu_s_val = E_s_val/(2*(1+nu_s_val))
    lambda_s_val = nu_s_val*2.*mu_s_val/(1. - 2.*nu_s_val)

    default_variables.update(dict(
        T=0.1,                 # Simulation end time [s]
        dt=0.0005,                # Time step size [s]
        recompute=30,           # Recompute the Jacobian matix within time steps                                                                                    
        recompute_tstep=10,     # Recompute the Jacobian matix over time steps (dangerous!)  
        folder="tube_2d",       # Output folder
        theta=0.50+0.0005,        # Shifted Crank-Nicolson scheme
        rho_f=025E3,              # Fluid density [g/cm3]
        mu_f=3.5E-3,              # Fluid dynamic viscosity [poise]
        rho_s=1.0E3,              # Solid density [g/cm3]
        mu_s=mu_s_val,          # Solid shear modulus or 2nd Lame Coef. 
        nu_s=nu_s_val,          # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid Young's modulus 
        dx_f_id=7,              # ID of marker in the fluid domain
        dx_s_id=8,              # ID of marker in the solid domain
        checkpoint_step=50,     # Save frequency of checkpoint files 
        save_step=1,            # Save frequency of files for visualisation
        save_deg=1,
        d_deg = 2,
        v_deg = 2,
        P_deg = 1,
        t_start = 2e-6,         # Start time of the impulse
        t_end = 2.1e-6,           # End time of the impulse
    ))

    return default_variables


def get_mesh_domain_and_boundaries(**namespace):
    # Import mesh file
    mesh = Mesh()
    with XDMFFile("mesh/tube_2d/tube_2d.xdmf") as infile:
        infile.read(mesh)
    #  # Rescale the mesh coordinated from [mm] to [m]
    # x = mesh.coordinates()
    # scaling_factor = 0.001  # from mm to m
    # x[:, :] *= scaling_factor
    # mesh.bounding_box_tree().build(mesh)
    # Import mesh boundaries
    boundaries = MeshValueCollection("size_t", mesh, 1) 
    with XDMFFile("mesh/tube_2d/tube_2d_facet.xdmf") as infile:
        infile.read(boundaries, "name_to_read")

    boundaries = cpp.mesh.MeshFunctionSizet(mesh, boundaries)
    
    # Define mesh domains
    domains = MeshValueCollection("size_t", mesh, 2) 
    with XDMFFile("mesh/tube_2d/tube_2d.xdmf") as infile:
        infile.read(domains, "name_to_read")

    domains = cpp.mesh.MeshFunctionSizet(mesh, domains)

    info_blue("Obtained mesh, domains and boundaries.")

    return mesh, domains, boundaries


class PressureImpulse(UserExpression):
    def __init__(self, force_val, t_start,t_end,t,**kwargs):
        self.force_val = force_val
        self.t_start = t_start
        self.t_end = t_end
        self.force = 0.0
        self.t = t

        super().__init__(**kwargs)

    def update(self, t):
        self.t = t

    def eval(self, value,x):
        if self.t >= self.t_start and self.t<=self.t_end:
            self.force = self.force_val 
        else:
            self.force = 0.0
        value[0] = self.force
        value[1] = 0

    def value_shape(self):
        return (2,)
    
# def initiate(dvp_, DVP, mesh, boundaries, **namespace):
#     # get the dofs on the fluid inlet boundary
#     # V = FunctionSpace(mesh, 'CG', 1)
#     # bc = DirichletBC(V, Constant(0), boundaries, 1)
#     # bdry_dofs = np.array(list(bc.get_boundary_values().keys())) 
#     # p_n = dvp_["n"].sub(2, deepcopy=True) 
#     # p_n.vector()[bdry_dofs] = 5e3
#     # bc = DirichletBC(DVP.sub(2), p_n, boundaries, 1)
    
#     # bc.apply(dvp_["n"].vector())

#     # return dict(dvp_=dvp_)
#     pass

def create_bcs(DVP, boundaries, F_fluid_linear, psi, mesh, **namespace):

    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    # Solid displacement BCs / x-component of displacement at solid inlet and outlet is 0
    d_s_inlet = DirichletBC(DVP.sub(0).sub(0), Constant(0), boundaries, 4)
    d_s_outlet = DirichletBC(DVP.sub(0).sub(0), Constant(0), boundaries, 5)

    

    # Fluid displacement BCs / displacement at fluid inlet and outlet is 0 
    d_f_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 1)
    d_f_outlet = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 2)

    # Fluid velocity BCs / no-slip at solid walls
    inflow_profile = ('4*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
    u_f_inlet = DirichletBC(DVP.sub(1), Expression(inflow_profile, degree=2), boundaries, 1)
    u_f_walls = DirichletBC(DVP.sub(1),  ((0.0, 0.0)), boundaries, 3)
    # ds for fluid inlet 
    # ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    # impulse_force = PressureImpulse(force_val=50, t_start=0.005, t_end=0.02, t=0.0)
    # F_fluid_linear -= inner(impulse_force, psi)*ds(1)

    # V = FunctionSpace(mesh, 'CG', 1)
    # bc = DirichletBC(V, Constant(0), boundaries, 1)
    # bdry_dofs = np.array(list(bc.get_boundary_values().keys())) 
    # p_n = dvp_["n"].sub(2, deepcopy=True) 
    # p_n.vector()[bdry_dofs] = 5e3
    # bcp = DirichletBC(DVP.sub(2), p_n, boundaries, 1)
    # bcp_wall = DirichletBC(DVP.sub(2), Constant(0), boundaries, 3)
    bcp_out = DirichletBC(DVP.sub(2), Constant(0), boundaries, 2)
    # Assemble boundary conditions
    bcs = [d_s_inlet, d_s_outlet, d_f_inlet, d_f_outlet, u_f_walls, u_f_inlet, bcp_out]

    # return dict(bcs=bcs, F_fluid_linear=F_fluid_linear, impulse_force=impulse_force)
    return dict(bcs=bcs)


# def pre_solve(t, impulse_force, **namespace):
#     # Update the time variable used for the inlet boundary condition
#     impulse_force.update(t)
#     return dict(impulse_force=impulse_force)
