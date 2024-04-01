#include "fem/bilinearform.hpp"
#include "fem/bilininteg.hpp"
#include "fem/coefficient.hpp"
#include "fem/fe_coll.hpp"
#include "fem/fespace.hpp"
#include "fem/gridfunc.hpp"
#include "fem/linearform.hpp"
#include "fem/lininteg.hpp"
#include "general/array.hpp"
#include "general/optparser.hpp"
#include "linalg/solvers.hpp"
#include "linalg/sparsemat.hpp"
#include "linalg/sparsesmoothers.hpp"
#include "linalg/vector.hpp"
#include "mfem.hpp"

#include "fstream"
#include "iostream"
#include <string>


using std::string;

int main(int argc, char *argv[])
{
    string mesh_file = "../data/star.mesh";
    int order = 1;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
    args.ParseCheck();

    mfem::Mesh mesh(mesh_file);
    mesh.UniformRefinement();

    mfem::H1_FECollection fec(order, mesh.Dimension());
    mfem::FiniteElementSpace fes(&mesh, &fec);

    std::cout << "Number of unknowns: " << fes.GetTrueVSize() << std::endl;

    mfem::Array<int> boundary_dofs;
    fes.GetBoundaryTrueDofs(boundary_dofs);

    mfem::GridFunction x(&fes);
    x=0.0;

    const double diffusion_coef = 10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    .0;

    mfem::ConstantCoefficient coef(diffusion_coef);

    mfem::LinearForm b(&fes);
    b.AddDomainIntegrator(new mfem::DomainLFIntegrator(coef));
    b.Assemble();

    mfem::BilinearForm a(&fes);
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
    a.Assemble();

    mfem::SparseMatrix A;
    mfem::Vector B, X;
    a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

    mfem::GSSmoother M(A);
    mfem::PCG(A, M, B, X);

    a.RecoverFEMSolution(X, b, x);
    x.Save("sol.gf");
    mesh.Save("mesh.mesh");
}