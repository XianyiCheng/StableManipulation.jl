using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Plots
using Convex, SCS
using JLD
using StableManipulation

include("../models/box_ground_frictional.jl")

dynamics! = BoxWorld.ode_dynamics!
conditions = BoxWorld.ode_conditions
affect! = BoxWorld.ode_affect!
affect_neg! = BoxWorld.ode_affect_neg!

domain = BoxWorld.domain
guard_set = BoxWorld.guard_set
jumpmap = BoxWorld.jumpmap

n_contacts = BoxWorld.n_contacts
Δt = BoxWorld.Δt
modes = BoxWorld.modes

tol_c = BoxWorld.tol_c

xref = [0; BoxWorld.h/2; 0; 1; 0; 0] # sliding to the right at velocity of 1
uref = [BoxWorld.μ*BoxWorld.m*BoxWorld.g; 0; 0] # reference body wrench
nominal_mode = [1 1 1 1] # right sliding mode

pusher_p = [-BoxWorld.w/2;0] # pusher_location

function pusher_box_discrete_dynamics(x, pusher_u, mode)
    Jc = [1 0; 0 1; -pusher_p[2] pusher_p[1]]
    u = Jc*pusher_u
    xn = BoxWorld.discrete_dynamics(x, u, mode)
    return xn
end

function pusher_box_dynamics(x, pusher_u, mode)
    Jc = [1 0; 0 1; -pusher_p[2] pusher_p[1]]
    u = Jc*pusher_u
    dx = BoxWorld.continuous_dynamics_differentiable(x, u, mode)
    return dx
end

pusher_u_ref = [uref[1]; 0]

# Linearize over xref and uref
A = ForwardDiff.jacobian(_x->pusher_box_dynamics(_x, pusher_u_ref, nominal_mode), xref)
B = ForwardDiff.jacobian(_u->pusher_box_dynamics(xref, _u, nominal_mode), pusher_u_ref)

# solver = () -> SCS.Optimizer(verbose=true)

# Q = Semidefinite(6)
# Y = Variable((2,6))
# I_Q = Matrix(I, 6, 6)

# prob = maximize(sum(Q) + tr(Q)) # TODO find some good cost 
# #prob.constraints += isposdef( Q )
# prob.constraints += isposdef( -(A*Q + Q*A' + B*Y + (B*Y)'))

# Convex.solve!(prob,solver)

# Q_sol = evaluate(Q)
# Q_sol[abs.(Q_sol) .< 1e-6] .= 0
# Y_sol = evaluate(Y)
# println(IOContext(stdout, :compact => true),"Q\n", Q_sol)
# println(IOContext(stdout, :compact => true), "Y\n",Y_sol)

# P = inv(0.5*(Q_sol + Q_sol'))
# K = Y_sol*P

# println("eigvals of Q_sol\n",eigvals(Q_sol + Q_sol'))
# JV = -(A*Q_sol + Q_sol*A' + B*Y_sol + (B*Y_sol)')
# println(isposdef(P))
# println(isposdef(JV + JV'))

# println(IOContext(stdout, :compact => true),"P\n", P)
# println(IOContext(stdout, :compact => true), "K\n",K)
