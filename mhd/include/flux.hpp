#pragma once

#include "mfem.hpp"

using namespace mfem;

extern const int num_equation;
extern const double polytropical_ratio;
extern const double Ms;
extern const double Ma;
extern const double Re;
extern const double Rem;

class RiemannSolver
{
    private:
        Vector flux1;
        Vector flux2;

    public:
        RiemannSolver();
        double Eval(const Vector &state1, const Vector &state2,
                    const Vector &nor, Vector &flux);
};

class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   RiemannSolver rsolver;
   Vector shape1;
   Vector shape2;
   Vector funval1;
   Vector funval2;
   Vector nor;
   Vector fluxN;

public:
   FaceIntegrator(RiemannSolver &rsolver_, const int dim);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

// Physicality check (at end)
bool StateIsPhysical(const Vector &state, const int dim);

// Pressure (EOS) computation
inline double ComputePressure(const Vector &state, int dim)
{
   const double den = state(0);

   return pow(den, polytropical_ratio) / (polytropical_ratio * Ms * Ms);
}

// Compute the vector flux F(u)
void ComputeFlux(const Vector &state, int dim, DenseMatrix &flux);

// Compute the scalar F(u).n
void ComputeFluxDotN(const Vector &state, const Vector &nor,
                     Vector &fluxN);

// Compute the maximum characteristic speed.
inline double ComputeMaxCharSpeed(const Vector &state, const int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const Vector B(state.GetData() + 1 + dim, dim);

   double B_sqr = 0;
   for (int d = 0; d < dim; d++) { B_sqr += B(d) * B(d); } 

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   const double denum = 1.0 / (Ma * sqrt(den));
   const double b = sqrt(B_sqr) * denum;

   // TODO: check if it's right
   const double bx = 0.;
   // const double bx = b;

   const double sound = sqrt(polytropical_ratio * pow(den, polytropical_ratio - 1.0));
   const double vel = sqrt(den_vel2 / den);

   const double A_ = sound*sound + b*b;
   const double fast_wave = sqrt(0.5 * (A_ + sqrt(A_*A_ - 4*sound*sound*bx*bx)));
   return (vel + sound);
}
