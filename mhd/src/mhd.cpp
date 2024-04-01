#include "mfem.hpp"
#include "flux.hpp"
#include "initial.hpp"
#include "operator.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

const int num_equation = 7;
const double polytropical_ratio = 5.0/ 3.0;
const double Ms = 1.0;
const double Ma = 1.0;
const double Re = 1.0;
const double Rem = 1.0;

double max_char_speed;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 1;

   int precision = 8;
   std::cout.precision(precision);

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   // 2. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   mfem::Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   MFEM_ASSERT(dim == 2, "Need a two-dimensional mesh for the problem definition");

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   mfem::ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new mfem::ForwardEulerSolver; break;
      case 2: ode_solver = new mfem::RK2Solver(1.0); break;
      case 3: ode_solver = new mfem::RK3SSPSolver; break;
      case 4: ode_solver = new mfem::RK4Solver; break;
      case 6: ode_solver = new mfem::RK6Solver; break;
      default:
         std::cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   mfem::DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   mfem::FiniteElementSpace fes(&mesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   mfem::FiniteElementSpace dfes(&mesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   mfem::FiniteElementSpace vfes(&mesh, &fec, num_equation, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == mfem::Ordering::byNODES, "");

   std::cout << "Number of unknowns: " << vfes.GetVSize() << std::endl;

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // The solution u has components {density, x-momentum, y-momentum, energy}.
   // These are stored contiguously in the BlockVector u_block.
   mfem::Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) { offsets[k] = k * vfes.GetNDofs(); }
   mfem::BlockVector u_block(offsets);

   // Momentum grid function on dfes for visualization.
   mfem::GridFunction mom(&dfes, u_block.GetData() + offsets[0]);

   // Initialize the state.
   mfem::VectorFunctionCoefficient u0(num_equation, InitialCondition);
   mfem::GridFunction sol(&vfes, u_block.GetData());
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      std::ofstream mesh_ofs("vortex.mesh");
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equation; k++)
      {
         mfem::GridFunction uk(&fes, u_block.GetBlock(k));
         std::ostringstream sol_name;
         sol_name << "vortex-" << k << "-init.gf";
         std::ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 7. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   mfem::MixedBilinearForm Aflux(&dfes, &fes);
   Aflux.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator()));
   Aflux.Assemble();

   mfem::NonlinearForm A(&vfes);
   RiemannSolver rsolver;
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim));

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution euler(vfes, A, Aflux.SpMat());

   // Visualize the density
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         std::cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << std::endl;
         visualization = false;
         std::cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << mom;
         sout << "pause\n";
         sout << std::flush;
         std::cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // Determine the minimum element size.
   double hmin = 0.0;
   if (cfl > 0)
   {
      hmin = mesh.GetElementSize(0, 1);
      for (int i = 1; i < mesh.GetNE(); i++)
      {
         hmin = std::min(mesh.GetElementSize(i, 1), hmin);
      }
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   double t = 0.0;
   euler.SetTime(t);
   ode_solver->Init(euler);

   if (cfl > 0)
   {
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      Vector z(A.Width());
      max_char_speed = 0.;
      A.Mult(sol, z);
      // dt = cfl * hmin / max_char_speed / (2*order+1);
      dt = 0.002;
   }

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = std::min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0)
      {
         // dt = cfl * hmin / max_char_speed / (2*order+1);
         dt = 0.002;
      }
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         std::cout << "time step: " << ti << ", time: " << t << ", dt: " 
            << cfl * hmin / max_char_speed / (2*order+1) << std::endl;
         if (visualization)
         {
            sout << "solution\n" << mesh << mom << std::flush;
         }
      }
   }

   tic_toc.Stop();
   std::cout << " done, " << tic_toc.RealTime() << "s." << std::endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m vortex.mesh -g vortex-1-final.gf".
   for (int k = 0; k < num_equation; k++)
   {
      GridFunction uk(&fes, u_block.GetBlock(k));
      std::ostringstream sol_name;
      sol_name << "vortex-" << k << "-final.gf";
      std::ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
   }

   // 10. Compute the L2 solution error summed for all components.
   if (t_final == 2.0)
   {
      const double error = sol.ComputeLpError(2, u0);
      std::cout << "Solution error: " << error << std::endl;
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}
