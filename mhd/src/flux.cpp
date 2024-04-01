#include "flux.hpp"

// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver() :
   flux1(num_equation),
   flux2(num_equation) { }

double RiemannSolver::Eval(const Vector &state1, const Vector &state2,
                           const Vector &nor, Vector &flux)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();

   MFEM_ASSERT(StateIsPhysical(state1, dim), "");
   MFEM_ASSERT(StateIsPhysical(state2, dim), "");

   const double maxE1 = ComputeMaxCharSpeed(state1, dim);
   const double maxE2 = ComputeMaxCharSpeed(state2, dim);

   const double maxE = std::max(maxE1, maxE2);

   ComputeFluxDotN(state1, nor, flux1);
   ComputeFluxDotN(state2, nor, flux2);

   double normag = 0;
   for (int i = 0; i < dim; i++)
   {
      normag += nor(i) * nor(i);
   }
   normag = sqrt(normag);

   for (int i = 0; i < num_equation; i++)
   {
      flux(i) = 0.5 * (flux1(i) + flux2(i))
                - 0.5 * maxE * (state2(i) - state1(i)) * normag;
   }

   return maxE;
}

void ComputeFlux(const Vector &state, int dim, DenseMatrix &flux)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   Vector vel(dim);
   for (int i = 0; i < dim; ++i)
      vel(i) = den_vel(i) / den;

   const Vector B(state.GetData() + 1 + dim, dim);

   double B_sqr = 0.0;
   for (int i = 0; i < dim; ++i)
      B_sqr += B(i)*B(i);

   // TODO: make inv_MA_sqr be a class field

   const double inv_Ma_sqr = 0.;
   // const double inv_Ma_sqr = 1.0 / (Ma*Ma);
   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   const double pres = ComputePressure(state, dim);

   for (int d = 0; d < dim; d++)
   {
      flux(0, d) = den_vel(d);
      for (int i = 0; i < dim; i++)
      {
         flux(1+i, d) = den_vel(i) * den_vel(d) / den - inv_Ma_sqr*B(i)*B(d);

         flux(1+dim+i, d) = vel(d)*B(i) - vel(i)*B(d);
      }
      flux(1+d, d) += pres + 0.5*inv_Ma_sqr*B_sqr;

   }
}

// Compute the scalar F(u).n
void ComputeFluxDotN(const Vector &state, const Vector &nor,
                     Vector &fluxN)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const Vector B(state.GetData() + 1 + dim, dim);

   double B_sqr = 0.0;
   for (int i = 0; i < dim; ++i)
      B_sqr += B(i)*B(i);
   
   const double inv_Ma_sqr = 1.0 / (Ma*Ma);
   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   const double pres = ComputePressure(state, dim);

   double den_velN = 0;
   for (int d = 0; d < dim; d++) { den_velN += den_vel(d) * nor(d); }

   double bN = 0;
   for (int d = 0; d < dim; d++) { bN += B(d) * nor(d); }

   fluxN(0) = den_velN;
   for (int d = 0; d < dim; d++)
   {
      fluxN(1+d) = den_velN * den_vel(d) / den + pres * nor(d) + (pres + 0.5*inv_Ma_sqr*B_sqr)* nor(d)
         - inv_Ma_sqr*bN*B(d);

      fluxN(1+dim+d) = (den_vel(d)*bN - den_velN*B(d))/ den;
   }
}