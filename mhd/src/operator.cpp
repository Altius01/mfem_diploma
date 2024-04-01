#include "operator.hpp"
#include "flux.hpp"

extern const int num_equation;
extern const double polytropical_ratio;
extern const double Ms;
extern const double Ma;
extern const double Re;
extern const double Rem;

FE_Evolution::FE_Evolution(FiniteElementSpace &vfes_,
                           Operator &A_, SparseMatrix &Aflux_)
   : TimeDependentOperator(A_.Height()),
     dim(vfes_.GetFE(0)->GetDim()),
     vfes(vfes_),
     A(A_),
     Aflux(Aflux_),
     Me_inv(vfes.GetFE(0)->GetDof(), vfes.GetFE(0)->GetDof(), vfes.GetNE()),
     state(num_equation),
     f(num_equation, dim),
     flux(vfes.GetNDofs(), dim, num_equation),
     z(A.Height())
{
   // Standard local assembly and inversion for energy mass matrices.
   const int dof = vfes.GetFE(0)->GetDof();
   DenseMatrix Me(dof);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi;
   for (int i = 0; i < vfes.GetNE(); i++)
   {
      mi.AssembleElementMatrix(*vfes.GetFE(i), *vfes.GetElementTransformation(i), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(i));
   }
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed = 0.;

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A.Mult(x, z);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by interpolating
   //     at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed bilinear form for
   //     each of the num_equation, computing (F(u), grad(w)) for each equation.

   DenseMatrix xmat(x.GetData(), vfes.GetNDofs(), num_equation);
   GetFlux(xmat, flux);

   for (int k = 0; k < num_equation; k++)
   {
      Vector fk(flux(k).GetData(), dim * vfes.GetNDofs());
      Vector zk(z.GetData() + k * vfes.GetNDofs(), vfes.GetNDofs());
      Aflux.AddMult(fk, zk);
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation);

   for (int i = 0; i < vfes.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes.GetElementVDofs(i, vdofs);
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation);
      mfem::Mult(Me_inv(i), zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

// Compute the flux at solution nodes.
void FE_Evolution::GetFlux(const DenseMatrix &x_, DenseTensor &flux_) const
{
   const int flux_dof = flux_.SizeI();
   const int flux_dim = flux_.SizeJ();

   for (int i = 0; i < flux_dof; i++)
   {
      for (int k = 0; k < num_equation; k++) { state(k) = x_(i, k); }
      ComputeFlux(state, flux_dim, f);

      for (int d = 0; d < flux_dim; d++)
      {
         for (int k = 0; k < num_equation; k++)
         {
            flux_(i, d, k) = f(k, d);
         }
      }

      // Update max char speed
      const double mcs = ComputeMaxCharSpeed(state, flux_dim);
      if (mcs > max_char_speed) { max_char_speed = mcs; }
      // else {cout << "Max char speed: " << mcs << endl;}
   }
}

bool StateIsPhysical(const Vector &state, const int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const Vector B(state.GetData() + 1 + dim, dim);

   if (den < 0)
   {
      std::cout << "Negative density: ";
      for (int i = 0; i < state.Size(); i++)
      {
         std::cout << state(i) << " ";
      }
      std::cout << std::endl;
      return false;
   }
   
   return true;
}
