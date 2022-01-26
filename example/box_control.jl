using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Plots
using Convex, SCS
using JLD
using StableManipulation

# the model includes: TODO
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


# LQR controller

# compute infinite horizon K
A_lqr = ForwardDiff.jacobian(_x->pusher_box_discrete_dynamics(_x, pusher_u_ref, nominal_mode), xref)
B_lqr = ForwardDiff.jacobian(_u->pusher_box_discrete_dynamics(xref, _u, nominal_mode), pusher_u_ref)
Q_lqr = Diagonal([0;1.0;1.0;1.0;0.1;1.0])
R_lqr = Diagonal([0.1;0.1])
Ks, _ = StableManipulation.riccati(A_lqr,B_lqr,Q_lqr,R_lqr,Q_lqr,50)
K_lqr = Ks[1]

function pusher_lqr_controller(x, contactMode)
    Jc = [1 0; 0 1; -pusher_p[2] pusher_p[1]]
    u = pusher_u_ref .- K_lqr*(x .- xref)
    
    # direct cutoff 
    u[u .> 20] .= 20
    u[u .< -20] .= -20
    # Note: there is no frictional force constraints

    f = Jc*u
    return f
end

# Lyapunov function
function V(x)
    q = x[1:3] + x[4:6]
    dq = x[4:6]

    v = 0.5*([q;dq] .- xref)'*Q_lqr*([q;dq] .- xref)

    return v
end 

function dVdx(x)
    res = ForwardDiff.gradient(V, x)
    return res
end

# TODO: hybrid controller

function hybrid_constraints_matrix(x, u_ref)
    # A*z + b>= 0
    # u = z[1:2]
    # β = z[3:175]
    # α = z[176:end]
    
    n = 2+173+19
    m = 19 
    A = zeros(m, n)
    b = zeros(m)

    # α = 10
    
    Vx = dVdx(x)
    Vv = V(x)
    
    n_modes = size(modes,1)
    
    β_idx = 1
    
    nominal_mode = modes[10,:] # [1 1 1 1]
    

    # mode dynamics
    
    for k = 1:n_modes
    
        m = modes[k,:]
        
        d_ineq, d_eq = domain(x, m)
        
        # TODO: debug this!!!
        if sum(d_ineq.<0)>0 || sum(abs.(d_eq) .> tol_c)>0
            b[k] = 0.1
            continue
        end
        
        n_ineq = size(d_ineq, 1)
        n_eq = size(d_eq, 1)
        
        dfdu = ForwardDiff.jacobian(_u->pusher_box_dynamics(x, _u, m), u_ref)
        
        b[k] = -Vx'*pusher_box_dynamics(x, u_ref, m)
        # α
        A[k, 175+k] = -Vv
        A[k, 1:2] = -Vx'*dfdu
        A[k, β_idx:β_idx+n_ineq-1] .= -d_ineq
        A[k, β_idx+n_ineq:β_idx+n_ineq+n_eq-1] .= -d_eq 
        A[k, β_idx+n_ineq+n_eq:β_idx+n_ineq+2*n_eq-1] .= d_eq 
        
        β_idx += n_ineq + 2*n_eq
        
    end
    
    
    # TODO: nominal_mode isn't always the last mode
    for k = 1:(n_modes-1)
        
        mode_from = modes[k,:]
        
        d_ineq, d_eq = guard_set(x, mode_from, nominal_mode)
        
        if sum(d_ineq.<0)>0 || sum(abs.(d_eq) .> tol_c)>0
            b[k] = 0.1
            continue
        end
        
        n_ineq = size(d_ineq, 1)
        n_eq = size(d_eq, 1)
        
        xp = jumpmap(x, mode_from, nominal_mode)
        
        dfdu = ForwardDiff.jacobian(_u->pusher_box_dynamics(xp, _u, nominal_mode), u_ref)
        
        b[n_modes+k] = -Vx'*pusher_box_dynamics(xp, u_ref, nominal_mode)
        A[k, 175+n_modes+k] = -Vv
        A[n_modes+k, 1:2] = -Vx'*dfdu
        A[n_modes+k, β_idx:β_idx+n_ineq-1] .= -d_ineq
        A[n_modes+k, β_idx+n_ineq:β_idx+n_ineq+n_eq-1] .= -d_eq 
        A[n_modes+k, β_idx+n_ineq+n_eq:β_idx+n_ineq+2*n_eq-1] .= d_eq 
        
        β_idx += n_ineq + 2*n_eq
        
    end
    
    return A, b
    
end

function hybrid_controller_reference(x, mode)
    u = pusher_u_ref .- K_lqr*(x .- xref)
    # direct cutoff 
    u[u .> 20] .= 20
    u[u .< -20] .= -20
    # Note: there is no frictional force constraints
    return u
end

function hybrid_controller(x, contactMode)
    
    u_ref = hybrid_controller_reference(x, contactMode) # from reference controller
    
    α_ref = 15

    z = Variable(194)
    A, b = hybrid_constraints_matrix(x, u_ref)
    
    μ_mnp = 0.8
    FC = [1 0; μ_mnp -1; μ_mnp 1]

    problem = minimize(sumsquares(z - [zeros(175); α_ref*ones(19)]))
    problem.constraints += A*z + b >= 0
    problem.constraints += z[3:end] >= 0
    #problem.constraints += FC*(z[1:2]+u_ref) >= 0
    problem.constraints += z[1:2] <= 20
    problem.constraints += z[1:2] >= -20


    Convex.solve!(problem, () -> SCS.Optimizer(verbose=false))
    
    z_sol = evaluate(z)
    if any(isnan.(z_sol)) || sum(z_sol) == Inf # infeasible
        z_sol = zeros(175)
#         println("Infeasible: ")
#         println(x)
#         println(contactMode)
    end
    
    u = z_sol[1:2] .+ u_ref
    
    # return space wrench
    Jc = [1 0; 0 1; -pusher_p[2] pusher_p[1]]
    f = Jc*u
    
    return f
    
end

print(hybrid_controller([0;1;0.3;0;0;0], [0,0,0,0]))

# Simulation

θ0 = -0.4
x0 = [0;BoxWorld.h/2 - sin(θ0)*BoxWorld.w/2 + 0.1;θ0;0;0;0]

controller = pusher_lqr_controller

initial_mode = BoxWorld.mode(x0, zeros(3))

tspan = (0.0, 3.0)
callback_length = 5*n_contacts

h_control = Δt

prob = ODEProblem(dynamics!, x0, tspan, (initial_mode, controller, [0.0], h_control, zeros(3)))
cb = VectorContinuousCallback(conditions, affect!, affect_neg!, callback_length)
sol = solve(prob, Tsit5(); callback = cb, abstol=1e-15,reltol=1e-15, adaptive=false,dt=Δt)
println("Simulation status: ", sol.retcode)

# Animation

n = length(sol.t)
# n = Int(floor(0.77/Δt))
x = zeros(n)
y = zeros(n)
θ = zeros(n)
for k = 1:n
    x[k] = sol.u[k][1]
    y[k] = sol.u[k][2]
    θ[k] = sol.u[k][3]
end

BoxWorld.animation(x,y,θ,n)