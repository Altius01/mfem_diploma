#pragma once

#include "mfem.hpp"

extern const double polytropical_ratio;

void InitialCondition(const mfem::Vector &x, mfem::Vector &y);