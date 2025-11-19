using Optim 
using LinearAlgebra
using Optim

# --- types: Julia 1.x style ---
abstract type Constraint end
abstract type ConvexConstraint <: Constraint end

struct BoxConstraint <: ConvexConstraint
    lower::AbstractVector
    upper::AbstractVector
end

struct BallConstraint <: ConvexConstraint
    center::AbstractVector   # center
    radius::Real             # radius (scalar)
    p::Real                  # l^p norm exponent
end

struct EqualityConstraint <: Constraint
    c::Function
    Jc!::Function
    cJc!::Function
end

function EqualityConstraint(c::Function, Jc!::Function)
    function cJc!(x::AbstractVector, storage::AbstractMatrix)
        Jc!(x, storage)
        return c(x)
    end
    return EqualityConstraint(c, Jc!, cJc!)
end

# --- Augmented Lagrangian object ---
mutable struct AugmentedLagrangian{Tl}
   F            # note: see TODO about Optim types below
   C::EqualityConstraint
   lambda::Tl
   mu::Real
   Dc::Matrix{Float64}
end

const AL = AugmentedLagrangian

function evaluate(x::AbstractVector, al::AL)
   c = al.C.c(x)
   return al.F.f(x) - dot(c, al.lambda) + (0.5*al.mu) * sum(abs2, c)
end

# gradient helpers (mutating out vector)
function gradient!(out::AbstractVector, x::AbstractVector, al::AL)
   eval_and_grad!(out, x, al)
   return out
end

function eval_and_grad!(out::AbstractVector, x::AbstractVector, al::AL)
    # compute constraint and Jacobian
    c = al.C.cJc!(x, al.Dc)

    # # tell Optim/NLSolversBase that the evaluation point changed
    # Optim.update!(al.F, x)

    # get function value
    f = Optim.value!(al.F, x)

    # get gradient
    Optim.gradient!(al.F, out)

    # add AL gradient contributions
    for i in axes(c, 1)
        @inbounds out .-= al.lambda[i] .* view(al.Dc, i, :)
        @inbounds out .+= (al.mu * c[i]) .* view(al.Dc, i, :)
    end

    # augmented Lagrangian value
    return f - dot(c, al.lambda) + (0.5 * al.mu) * sum(abs2, c)
end


gradient(x, al::AL) = gradient!(zeros(eltype(x), length(x)), x, al)


#
# this is a trivial first write-up of NW, Algorithm 17.4
# with the new `update!` mechanism, I'd like to incorporate the
# Lagrange-multiplier and penalty updates into the minimisation
# procedure. But maybe this is not a good idea, to be tested.
#

function optimize( F::OnceDifferentiable,
                   C::EqualityConstraint,
                   x0::AbstractVector;
                   iterations = 5000,
                   c_tol = 1e-6,
                   g_tol = 1e-6,
                   verbose = 1,
                   Optimizer = Optim.ConjugateGradient )

   # initialise
   al = AugmentedLagrangian(F, C, x0)
   eta = al.mu^(-0.1)
   # TODO: this needs to distinguish whether F, C are OnceDifferentiable
   # or TwiceOnceDifferentiable!!!!
   ALobj = OnceDifferentiable( x_ -> evaluate(x_, al),
                                   (x_,g_) -> gradient!(g_, x_, al),
                                   x0)


   x = copy(x0)
   iteration = 0
   if verbose >= 1
      @show("  it  |     |c|_∞        |g_AL|_∞      μ         η  \n")
      @show("------|----------------------------------------------\n")
   end
   while iteration < iterations

      # solve the sub-problem
      options = OptimizationOptions(g_tol=g_tol, iterations=iterations - iteration, store_trace = true)
      result = Optim.optimize(ALobj, x, Optimizer(), options)
      # TODO: unclear what to do if this step fails? it could still be ok after
      # we update the lagrange multipliers? For now just print a warning and continue.
      if !Optim.converged(result)
         warn("an inner AL iteration has not converged")
         # return minimizer(result)
      end

      # check for convergence, perform update for multipliers and tolerances
      iteration += Optim.iterations(result)
      x = Optim.minimizer(result)
      nrm_g = Optim.g_norm_trace(result)[end]

      c = C.c(x)    # TODO: this is a superfluous call to C (but #282 would fix this)

      if verbose >= 1
         @show(" %4d |  %1.4e   %1.4e   %1.2e   %1.2e\n",
                  iteration, norm(x, Inf), nrm_g, al.mu, eta)
      end

      if norm(c, Inf) < c_tol
         return x, al
      elseif norm(c, Inf) <= eta
         al.lambda = al.lambda - al.mu * c
         eta /= al.mu^0.9  # tighten constraint tolerance
      else
         al.mu *= 100.0
         eta = al.mu^(-0.1)
      end
      if al.mu > 1e10
         warn("something horrible is happening")
         return x, al
      end

   end

   # iteration > iterations
   warn("too many iterations in `optimize` ")
   return x, al
end
@testset "Finite Difference Testing" begin
    f(x) = x[1]^2 + x[2]^2 + x[2]^4
    df(x) = [2*x[1], 2*x[2]+4*x[2]^3]
    c(x) = x[1]^2 + (x[2]-0.5)^2 - 2.0
    dc(x) = [ 2*x[1] 2*(x[2]-0.5) ]

    # the minimiser should be [0.0,0.0]
    x0 = [1.0, 0.0]


    F = OnceDifferentiable(f, (x,g) -> copy!(g, df(x)), x0)
    C = EqualityConstraint(c, (x,g) -> copy!(g, dc(x)) )


    println("------------------------------------------------------------")
    println("       Finite Difference Testing the AL ")
    @show("   h   |   err \n")
    @show("-------|-------------\n")
    # finite-difference test of the augmented Lagrangian implementation
    al = AugmentedLagrangian(F, C, x0)
    al.lambda = 0.0
    al.mu = 1.0
    A = evaluate(x0, al)
    dA = gradient(x0, al)
    err = Float64[]
    for p = 2:12
       h = 0.1^p
       dAh = zeros(length(dA))
       for n = 1:length(x0)
          x0[n] += h
          dAh[n] = (evaluate(x0, al) - A)  / h
          x0[n] -= h
       end
       push!(err, maximum(dA - dAh))
       @show(" 1e-%2d |  %1.4e \n", p, err[end])
    end
    @test minimum(err) < 1e-4 * err[1]
    if minimum(err) < 1e-4 * err[1]
       println("looks like the FD test has passed...")
    else
       warn("""the finite difference test for the augmented Lagrangien didn't
             pass; please check visually what happened and debug""")
    end


    println("------------------------------------------------------------")
    println("      Try to optimise something simple")
    x, al = optimize(F, C, x0)
    println("Converged to ", x, "; λ = ", al.lambda)
    println("First-order optimality: ")
    al.mu = 0.0; g = gradient(x, al); C = al.C.c(x)
    println("   ∇ₓL(x, λ) = ", g)
    println("        c(x) = ", C)
    println("  |∇L(x, λ)| = ", max(vecnorm(g), vecnorm(C)))
    @test vecnorm(g) < 1e-6
    @test vecnorm(C) < 1e-6
    end
