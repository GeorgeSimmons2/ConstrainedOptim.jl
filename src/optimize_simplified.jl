# Simplified optimization utilities
# The ProjectedGradientDescent and convex constraint optimization 
# require significant updates to work with modern Optim.jl
# For now, we focus on the Augmented Lagrangian method

struct ProjectedGradientDescent{T} <: AbstractOptimizer
    linesearch!::Function
    P::T
    precondprep!::Function
end

function ProjectedGradientDescent(; linesearch!::Function = nothing,
                P = nothing, precondprep! = (P, x) -> nothing)
    ProjectedGradientDescent(linesearch! === nothing ? backtracking_linesearch : linesearch!, P, precondprep!)
end

# Stub functions for convex constraint optimization
# Full implementation would require major refactoring for modern Optim.jl

project!(X, eq::ConvexConstraint) = nothing

function project!(X, eqc::BoxConstraint)
    @inbounds for (i, x) in enumerate(X)
        X[i] = max(min(x, eqc.upper[i]), eqc.lower[i])
    end
end

function project!(X, eqb::BallConstraint)
    @inbounds for (i, x) in enumerate(X)
        X[i] = x/max(eqb.radius, norm(x - eqb.center, eqb.p))
    end
end

# Overload to allow calling with convex constraints
# This is a stub - full implementation requires modern Optim.jl integration
function optimize(d, initial_x::Array, bc::ConvexConstraint, method::AbstractOptimizer, options::Optim.Options)
    error("Convex constrained optimization not yet fully implemented for modern Optim.jl. Use Augmented Lagrangian method instead.")
end

function optimize(d, initial_x::Array, bc::ConvexConstraint, method::AbstractOptimizer)
    optimize(d, initial_x, bc, method, Optim.Options())
end
