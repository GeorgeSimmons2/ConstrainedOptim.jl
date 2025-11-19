module ConstrainedOptim


using Optim
using Printf
using LinearAlgebra
import Optim: optimize, AbstractOptimizer

export EqualityConstraint, BoxConstraint, BallConstraint, ProjectedGradientDescent, optimize
include("types.jl")
include("objective_helpers.jl")
include("augmented_lagrangian.jl")
include("optimize.jl")
end # module
