# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from dolfin import assemble, derivative, TrialFunction, Matrix, norm, MPI, PETScOptions, as_backend_type
from petsc4py import PETSc

def solver_setup(F_fluid_linear, F_fluid_nonlinear, F_solid_linear, F_solid_nonlinear,
                 DVP, dvp_, up_sol, compiler_parameters, **namespace):
    """
    Pre-assemble the system of equations for the Jacobian matrix for the Newton solver
    """
    F_lin = F_fluid_linear + F_solid_linear
    F_nonlin = F_solid_nonlinear + F_fluid_nonlinear
    F = F_lin + F_nonlin

    chi = TrialFunction(DVP)
    J_linear = derivative(F_lin, dvp_["n"], chi)
    J_nonlinear = derivative(F_nonlin, dvp_["n"], chi)

    A_pre = assemble(J_linear, form_compiler_parameters=compiler_parameters,
                     keep_diagonal=True)
    A = Matrix(A_pre)
    b = None

    # Option not available in FEniCS 2018.1.0
    # up_sol.parameters['reuse_factorization'] = True

    return dict(F=F, J_nonlinear=J_nonlinear, A_pre=A_pre, A=A, b=b, up_sol=up_sol)


def newtonsolver(F, J_nonlinear, A_pre, A, b, bcs, lmbda, recompute, recompute_tstep, compiler_parameters,
                 dvp_, up_sol, dvp_res, rtol, atol, max_it, counter, first_step_num, verbose, **namespace):
    """
    Solve the non-linear system of equations with Newton scheme. The standard is to compute the Jacobian
    every time step, however this is computationally costly. We have therefore added two parameters for
    re-computing only every 'recompute' iteration, or for every 'recompute_tstep' time step. Setting 'recompute'
    to != 1 is faster, but can impact the convergence rate. Altering 'recompute_tstep' is considered an advanced option,
    and should be used with care.
    """
    # Initial values
    iter = 0
    residual = 10**8
    rel_res = 10**8

    # Capture if residual increases from last iteration
    last_rel_res = residual
    last_residual = rel_res

    # Initialize ksp solver.
    ksp = PETSc.KSP().create()
    ksp_viewer = PETSc.Viewer().createASCII("ksp_output.txt")
    pc_viewer = PETSc.Viewer().createASCII("pc_output.txt")
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps') # Default value "petsc" causes diverging solve
    ksp.setMonitor(lambda ksp, its, rnorm: print(f"KSP: {its} {rnorm}") if MPI.rank(MPI.comm_world) == 0 else None)
    ksp.setOperators(as_backend_type(A).mat())
    while rel_res > rtol and residual > atol and iter < max_it:
        # Check if recompute Jacobian from 'recompute_tstep' (time step)
        recompute_for_timestep = iter == 0 and (counter % recompute_tstep == 0)

        # Check if recompute Jacobian from 'recompute' (iteration)
        recompute_frequency = iter > 0 and iter % recompute == 0

        # Recompute Jacobian due to increased residual
        recompute_residual = iter > 0 and last_residual < residual

        # Recompute Jacobian on first step of simulation (important if restart is used)
        recompute_initialize = iter == 0 and counter == first_step_num

        if recompute_for_timestep or recompute_frequency or recompute_residual or recompute_initialize:
            if MPI.rank(MPI.comm_world) == 0 and verbose:
                print("Compute Jacobian matrix")
            q = id(A)
            assemble(J_nonlinear, tensor=A,
                         form_compiler_parameters=compiler_parameters,
                         keep_diagonal=True)
            t = id(A)
            if MPI.rank(MPI.comm_world) == 0:
                print(q, t)
            A.axpy(1.0, A_pre, True)
            A.ident_zeros()
            [bc.apply(A) for bc in bcs]
            # up_sol.set_operator(A)
            
    
        # Compute right hand side
        b = assemble(-F, tensor=b)

        # Apply boundary conditions and solve
        [bc.apply(b, dvp_["n"].vector()) for bc in bcs]
        
        # Solve linear system
        # up_sol.solve(dvp_res.vector(), b)
        if not as_backend_type(A).mat().assembled:
            print("Warning: Matrix not assembled")
            as_backend_type(A).mat().assemble()
        
        pc.setUp()
        pc.view(pc_viewer)
        pc_output = open("pc_output.txt", "r")
        if MPI.rank(MPI.comm_world) == 0:
            print(pc_output.read())
            pc_output.close()

        ksp.solve(as_backend_type(b).vec(), as_backend_type(dvp_res.vector().vec()))
        ksp.view(ksp_viewer)
        ksp_output = open("ksp_output.txt", "r")
        if MPI.rank(MPI.comm_world) == 0:
            print(ksp_output.read())
            ksp_output.close()
        assert ksp.getConvergedReason() > 0

        dvp_["n"].vector().axpy(lmbda, dvp_res.vector())
        [bc.apply(dvp_["n"].vector()) for bc in bcs]
        # print("dvp_res", dvp_res.vector().get_local())
        # Reset residuals
        last_residual = residual
        last_rel_res = rel_res

        # Check residual
        residual = b.norm('l2')
        rel_res = norm(dvp_res, 'l2')
        if rel_res > 1E20 or residual > 1E20:
            raise RuntimeError("Error: The simulation has diverged during the Newton solve.")

        if MPI.rank(MPI.comm_world) == 0 and verbose:
            print("Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) "
                  % (iter, residual, atol, rel_res, rtol))
        iter += 1

    return dict(up_sol=up_sol, A=A)