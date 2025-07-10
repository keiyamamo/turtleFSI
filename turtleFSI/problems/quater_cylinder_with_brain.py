# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import *
from turtleFSI.problems import *
import stress_strain as StrStr


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
        dt=0.1,       # Time step [s]
        checkpoint_step=1000, # Checkpoint frequency
        theta=1.0,     # Temporal scheme
        save_step=1,
        P_final=10000,  # Steady State pressure applied to wall
        t_start_p=0.0,  # Start time for pressure application
        t_end_p=0.9,    # End time for pressure application
        save_deg=2,
        recompute=20,
        recompute_tstep=110,
        robin_bc=False,
    
        # Problem specific
        dx_s_id=[1, 2],     # Id of the solid domain
        folder="quater_cylinder_wrapped",          # Folder to store the results
        fluid="no_fluid",                 # Do not solve for the fluid
        extrapolation="no_extrapolation",  # No displacement to extrapolate
        solid_vel=0.0003, # this is the velocity of the wall with prescribed displacement
        solid_properties=[{"dx_s_id": 1, "material_model": "StVenantKirchoff", "rho_s": 1.0E3, "mu_s": mu_s_val,
                           "lambda_s": lambda_s_val},
                          {"dx_s_id": 2, "material_model": "StVenantKirchoff", "rho_s": 1.0E3, "mu_s": mu_s_val_brain,
                           "lambda_s": lambda_s_val_brain}],
        ))   

    return default_variables


def get_mesh_domain_and_boundaries(dx_s_id, **namespace):

    mesh_path = "/Users/keiyamamoto/Documents/MyMesh/cylinder/quater_cylinder_with_brain_tissue/mesh.xdmf"
    
    mesh = Mesh()
    with XDMFFile(mesh_path) as infile:
        infile.read(mesh)

    mf_path = "/Users/keiyamamoto/Documents/MyMesh/cylinder/quater_cylinder_with_brain_tissue/mf.xdmf"
    # Import mesh boundaries
    boundaries = MeshValueCollection("size_t", mesh, 2) 
    with XDMFFile(mf_path) as infile:
        infile.read(boundaries, "name_to_read")

    boundaries = cpp.mesh.MeshFunctionSizet(mesh, boundaries)
    
    # Define mesh domains
    domains = MeshValueCollection("size_t", mesh, 3) 
    with XDMFFile(mesh_path) as infile:
        infile.read(domains, "name_to_read")

    domains = cpp.mesh.MeshFunctionSizet(mesh, domains)

    sph_rad = 0.0027
    i = 0
    for cell in cells(mesh):
        idx_cell = domains.array()[i]
        if idx_cell == dx_s_id[0]:
            mid = cell.midpoint()
            dist_sph_center = sqrt((mid.x() - 0.0) ** 2 + (mid.y() - 0.0) ** 2)
            if dist_sph_center > sph_rad:
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


def create_bcs(F_solid_linear, DVP, boundaries, P_final, t_start_p, t_end_p, mesh, psi, **namespace):
    
    p_out_bc_val = InnerP(t=0.0, t_start=t_start_p, t_end=t_end_p, P_final=P_final, degree=2)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    F_solid_linear += p_out_bc_val * inner(n, psi) * ds(5)

    # fix bottom surface in y-direction
    bc_bottom = DirichletBC(DVP.sub(0).sub(1), Constant(0.0), boundaries, 2)
    # fix x motion on the left wall
    bc_left = DirichletBC(DVP.sub(0).sub(0), Constant(0.0), boundaries, 4)
    # fix z motion on the inlet and outlet
    bc_inlet = DirichletBC(DVP.sub(0).sub(2), Constant(0.0), boundaries, 6)
    bc_outlet = DirichletBC(DVP.sub(0).sub(2), Constant(0.0), boundaries, 1)

    # completely fix the outer layer of the solid
    bc_outer = DirichletBC(DVP.sub(0), (0, 0, 0), boundaries, 3)  # outer layer

    bcs = [bc_bottom, bc_left, bc_inlet, bc_outlet, bc_outer]
    
    return dict(bcs=bcs, p_out_bc_val=p_out_bc_val, F_solid_linear=F_solid_linear)


def pre_solve(t, p_out_bc_val, **namespace):
    # Update the pressure boundary condition
    p_out_bc_val.update(t)
    
    return dict(p_out_bc_val=p_out_bc_val)
