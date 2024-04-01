#pragma once

#include "flux.hpp"
#include "operator.hpp"
#include "linalg/operator.hpp"
#include "mfem.hpp"

using namespace mfem;

extern double max_char_speed;



class FE_Evolution : public TimeDependentOperator 
{
    private:
        const int dim;

        FiniteElementSpace &vfes;
        Operator &A;
        SparseMatrix &Aflux;
        DenseTensor Me_inv;

        mutable Vector state;
        mutable DenseMatrix f;
        mutable DenseTensor flux;
        mutable Vector z;

        void GetFlux(const DenseMatrix &state_, DenseTensor &flux_) const;

    public:
        FE_Evolution(FiniteElementSpace &vfes_,
                Operator &A_, SparseMatrix &Aflux_);

        virtual void Mult(const Vector &x, Vector &y) const;

        virtual ~FE_Evolution() { }
};

bool StateIsPhysical(const Vector &state, const int dim);