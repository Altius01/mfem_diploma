#include "initial.hpp"

void InitialCondition(const mfem::Vector &x, mfem::Vector &y)
{
   MFEM_ASSERT(x.Size() == 2, "");

   const double vel_inf = 1.;
   const double den_inf = 1.;
   const double B_inf = 0.;

   // Derive remainder of background state from this and Minf
   const double pres_inf = pow(den_inf, polytropical_ratio);

   const double sq_pi = sqrt(4*M_PI);

   const double velX = vel_inf * -sin(2*M_PI*x(1));
   const double velY = vel_inf * sin(2*M_PI*x(0));
   const double vel2 = velX * velX + velY * velY;

   const double Bx = B_inf * -sin(2*M_PI*x(1));
   const double By = B_inf * sin(4*M_PI*x(0));

   const double den = den_inf;
   const double pres = pres_inf;

   const double shift = 100.;

   y(0) = shift + den;
   y(1) = shift + den * velX;
   y(2) = shift + den * velY;
   y(3) = shift + Bx;
   y(4) = shift + By;
}
