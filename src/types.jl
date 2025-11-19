abstract type Constraint end
abstract type ConvexConstraint <: Constraint end
struct BoxConstraint <: ConvexConstraint
    lower
    upper
end

struct BallConstraint <: ConvexConstraint
    center # center
    radius # radius
    p # l^p norm
end

struct EqualityConstraint <: Constraint
    c
    Jc!
    cJc!
end

function EqualityConstraint(c, Jc!)
    function cJc!(x::Array, storage::Array)
        Jc!(x, storage)
        return c(x)
    end
    return EqualityConstraint(c, Jc!, cJc!)
end

mutable struct AugmentedLagrangian{Tl<:Union{Real, Vector}}
   f::Function
   df::Function
   fdf::Function
   C::EqualityConstraint
   lambda::Tl
   mu::Real
   Dc::Matrix
end

const AL = AugmentedLagrangian
function AugmentedLagrangian(F::Any,
                 C::EqualityConstraint,
                 x0::AbstractVector;
                 lambda=:auto, mu=10.0)
   # TODO: maybe not ideal to evaluate F, C here, maybe better to set
   # dimensions when C is first called??? But see #282
   C0 = C.c(x0)
   dim_x = length(x0)
   dim_c = length(C0)
   if lambda == :auto
      lambda = - mu * C0
   end
   
   # Extract function methods from OnceDifferentiable or similar
   # If F is an OnceDifferentiable, use its methods
   f = x -> F.f(x)
   df = (g, x) -> F.df(g, x)
   fdf = (g, x) -> F.fdf(g, x)
   
   return AugmentedLagrangian(f, df, fdf, C, lambda, mu, zeros(dim_c, dim_x))
end
