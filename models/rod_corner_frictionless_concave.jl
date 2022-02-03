# this is a 2D model of a rod hitting a corner consisted of 2 frictionless walls 
# It includes: system properties, hybrid system description, functions for the ODE solver to simulate and animate the system

module RodWorld

using ForwardDiff
using Plots
using LinearAlgebra

debug = true

### (1) system properties & contact constraints

const g = 9.8
const m = 1

const l = 1.0 # length of the box
const M = [m 0 0; 0 m 0; 0 0 m*l*l/12]# inertia matrix

const modes = [false false ; # both free
               true false; # x = 0 wall in contact
               false true; # y = 0 wall in contact
               true true] # both in contact

const n_contacts = 2

# control step size
const Δt = 0.05
# tolerance for contacts
const tol_c = 5e-4

# 2D rotation matrix
function R_2D(θ)
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    return R
end

# constraints and contacts 
# TODO: debug this
# function compute_a(q)
#     θ = q[3]
#     p = R_2D(θ)*[-l/2;0] + q[1:2]
#     a = [p[1]+max(0,p[2]-tol_c); p[2] + max(0,p[1]-tol_c)] 
#     return a
# end

function compute_a(q)
    θ = q[3]
    p = R_2D(θ)*[-l/2;0] + q[1:2]
    if p[2] > 0
        a1 = sqrt(p[1]^2 + p[2]^2)
    else
        a1 = p[1]
    end
    if p[1] > 0
        a2 = sqrt(p[1]^2 + p[2]^2)
    else
        a2 = p[2]
    end
    return [a1;a2]
end

# constraints and contacts 
function compute_p(q)
    θ = q[3]
    a = R_2D(θ)*[-l/2;0] + q[1:2]

    return a
end

# function compute_A(q)
#     A = ForwardDiff.jacobian(compute_a, q)
#     return A
# end
function compute_A(q)
    A = ForwardDiff.jacobian(compute_p, q)
    return A
end

function compute_dA(q, dq)
    dA = reshape(ForwardDiff.jacobian(compute_A, q)*dq, n_contacts, 3)
    return dA
end

function contact_mode_constraints(x, contactMode)
    q = x[1:3]
    dq = x[4:6]
    
    #Compute A and dA at the current state, only select rows for the current contact mode
    A = compute_A(q)
    dA = compute_dA(q, dq)
    A = A[contactMode,:]
    dA = dA[contactMode,:]

    return A, dA
end

function contact_velocity_constraints(x, contactMode)
    # contact constraints on velocities (cones of velocities), including equalities and inequalities 
    # reture A_eq, A_geq, A_eq*dq = 0, A_geq*dq >= 0
    q = x[1:3]
    dq = x[4:6]

    A = compute_A(q)
    dA = compute_dA(q, dq)

    AA_eq = zeros(0,3)
    AA_geq = zeros(0,3)
    dAA_eq = zeros(0,3)
    dAA_geq = zeros(0,3)

    for k = 1:n_contacts
        if contactMode[k] == true
            AA_eq = [AA_eq; A[k,:]']
            dAA_eq = [dAA_eq; dA[k,:]']
        else
            AA_geq = [AA_geq; A[k,:]']
            dAA_geq = [dAA_geq; dA[k,:]']
        end
    end

    return AA_eq, AA_geq, dAA_eq, dAA_geq

end


function solveEOM(x, contactMode, u_control)
    # contactMode: bool vector, indicates which constraints are active
    q = x[1:3]
    dq = x[4:6]
    
    A, dA = contact_mode_constraints(x, contactMode)
    
    c = size(A,1)
    
    # compute EOM matrices
    N = [0; g; 0]
    C = zeros(3,3)
    # Y = zeros(3)
    Y = u_control
    
    #
    blockMat = [M A'; A zeros(size(A,1),size(A',2))] 

    b = [Y-N-C*dq; -dA*dq]
    
    H = [zeros(3,size(blockMat,2)); zeros(size(blockMat,1)-3,3) 1e-6*I]
    z =(blockMat.+H)\b

    # z = blockMat\b
    
    ddq = z[1:3]

    if (sum(contactMode)>=1)
        λ = z[4:end]
    else
        λ = []
    end
    
    return ddq, λ
end

function computeResetMap(x, contactMode)
    q = x[1:3]
    dq = x[4:6]

    A, dA = contact_mode_constraints(x, contactMode)
    
    c = size(A, 1)

    blockMat = [M A'; A zeros(size(A,1),size(A',2))] 

    # z = blockMat\[M*dq; zeros(c)]

    H = [zeros(3,size(blockMat,2)); zeros(size(blockMat,1)-3,3) 1e-6*I]
    z =(blockMat.+H)\[M*dq; zeros(c)]

    dq_p = z[1:3]
    p_hat = z[4:end]

    return dq_p, p_hat
end

function compute_FA(x, u_control)
    q = x[1:3]
    dq = x[4:6]
    
    a = compute_a(q)
    
    active_a = abs.(a) .< tol_c
    inactive_a = abs.(a) .> tol_c
    
    all_m = active_a #all-constraints-active mode
    contactMode = all_m
    possibleModes = zeros(Bool, 0, size(modes,2))
    
    for k = 1:size(modes,1)
        m = modes[k, :]
        if length(findall(z->z==true, m[inactive_a])) == 0
            possibleModes = [possibleModes; m']
        end
    end
    
    ddq_union, λ_union = solveEOM(x, all_m, u_control)
    
    active_idx = findall(z->z==true,all_m)
    all_m_active = all_m[active_idx]
    
    for kk = 1:size(possibleModes, 1)
        m = possibleModes[kk,:]
        ddq, λ = solveEOM(x, m, u_control)
        
        m_active = m[active_idx]
        
        if all(-λ.>=0) && all(-λ_union[m_active.!=all_m_active].<=0)
            contactMode = m
            break;
        end
    end
    
    return contactMode
end

function compute_IV(x)
    
    q = x[1:3]
    dq = x[4:6]
    
    a = compute_a(q)
    
    active_a = abs.(a) .< tol_c
    inactive_a = abs.(a) .> tol_c
    
    all_m = active_a #all-constraints-active mode
    contactMode = all_m
    possibleModes = zeros(Bool, 0, size(modes,2))
    
    for k = 1:size(modes,1)
        m = modes[k, :]
        if length(findall(z->z==true, m[inactive_a])) == 0
            possibleModes = [possibleModes; m']
        end
    end
    
    dq_p_union, p_hat_union = computeResetMap(x, all_m)
    
    active_idx = findall(z->z==true,all_m)
    all_m_active = all_m[active_idx]
    
    for kk = 1:size(possibleModes, 1)
        m = possibleModes[kk,:]
        m_active = m[active_idx]
        dq_p, p_hat = computeResetMap(x, m)
        
        if all(-p_hat.>=0) && all(-p_hat_union[m_active.!=all_m_active].<=0)
            contactMode = m
            break;
        end
    end
    
    return contactMode
end

function mode(x, u_control)
    return Vector{Bool}(compute_FA(x, u_control))
end

function guard_conditions(x, contactMode, u_control)
    q = x[1:3]
    dq = x[4:6]
    
    a = compute_a(q)
    a[(contactMode .== true) .& (a .< tol_c)] .= 0.0
    
    λ_all = zeros(length(a))
    
    ddq, λ = solveEOM(x, contactMode, u_control)
    
    λ_all[contactMode .== true] = λ
    
    c = [a; λ_all]
    
    dir = [-ones(Int,length(a)); ones(Int,length(λ_all))]
    
    return c, dir
end

### (2.2) wrapper for ode solver

function ode_dynamics!(dx, x, p, t)
    # p from integrator: (contact mode, controller, t_control, h_control, u_control)
    
    # bound the simulation
     # bound the simulation
     if any(abs.(x[1:2]).>5) || all(x[1:2].<0)
        dx .= zeros(6)
        return
    end

    q = x[1:3]
    dq = x[4:6]
    contactMode = p[1]
    controller = p[2]
    t_control = p[3][1]
    h_control = p[4]
    u_control = zeros(3)
    u_control .= p[5]
    
    if t > (t_control + h_control)
        p[3] .= [Float64(t)]
        u_control = controller(x, contactMode)
        p[5] .= u_control
    end

    new_contactMode = abs.(compute_a(x[1:3])) .< tol_c
    # println(new_contactMode)
    if any((new_contactMode.==false) .& contactMode)
        # new_contactMode = contactMode .| new_contactMode
        controller = p[2]
        p[1] .= reshape(new_contactMode,size(p[1]))
        u_control = controller(x, new_contactMode)
        p[5] .= u_control
        if debug == true
            println("New mode from geometry: ", new_contactMode)
            println(x)
        end
    end

    if all(new_contactMode) && ~all(new_contactMode .& contactMode)
        # new_contactMode = contactMode .| new_contactMode
        controller = p[2]
        p[1] .= reshape(new_contactMode,size(p[1]))
        u_control = controller(x, new_contactMode)
        p[5] .= u_control
        if debug == true
            println("New mode from geometry: ", new_contactMode)
            println(x)
        end
    end
    
    ddq, λ = solveEOM(x, contactMode, u_control)
    
    dx .= [dq; ddq]
end

function ode_conditions(out, x, t, integrator)
    contactMode = integrator.p[1]
    u_control = integrator.p[5]
    c, dir = guard_conditions(x, contactMode, u_control)
    out .= c
end

function ode_affect!(integrator, idx)
    # if debug == true
    #     println("upper crossing.")
    # end
    contactMode = integrator.p[1]
    x = integrator.u
    u_control = integrator.p[5]
    c, dir = guard_conditions(x, contactMode, u_control)
    
    # only consider upcrossing forces and constraints values(FA comp)
    # forces
    if dir[idx] > 0
        new_contactMode = compute_FA(x, u_control)
        controller = integrator.p[2]
        u_control = controller(x, new_contactMode)
        integrator.p[5] .= u_control
        if (eltype(new_contactMode)==Float64)
            return
        end
        integrator.p[1] .= reshape(new_contactMode,size(integrator.p[1]))
        if debug == true
            println("New mode from FA: ", new_contactMode)
            println(x, u_control)
        end
    else

        new_contactMode = abs.(compute_a(x[1:3])) .< tol_c
        # if any(new_contactMode.!=contactMode)
        #     # new_contactMode = contactMode .| new_contactMode
        #     controller = integrator.p[2]
        #     integrator.p[1] .= reshape(new_contactMode,size(integrator.p[1]))
        #     u_control = controller(x, new_contactMode)
        #     integrator.p[5] .= u_control
        #     if debug == true
        #         println("New mode from geometry: ", new_contactMode)
        #         println(x)
        #     end
        # end
    end
end

function ode_affect_neg!(integrator, idx)
    # if debug == true
    #     println("down crossing.")
    # end
    contactMode = integrator.p[1]
    x = integrator.u
    u_control = integrator.p[5]
    c, dir = guard_conditions(x, contactMode, u_control)
    
    # only consider down crossing constraint value(IV comp)
    if dir[idx] < 0
        new_contactMode = compute_IV(x)
        if (all(new_contactMode .==false))
            return
        end
        controller = integrator.p[2]
        u_control = controller(x, new_contactMode)
        integrator.p[5] .= u_control
        
        dq_p, p_hat = computeResetMap(x, new_contactMode)
        integrator.u .= [x[1:3]; dq_p]
        integrator.p[1] .= reshape(new_contactMode,size(integrator.p[1]))
        
        if debug == true
            println("New mode from IV: ", new_contactMode)
            println(x)
        end
    end
end

function rk4_step(f,xk,u)
    # classic rk4 parameters
    a21 = 0.5
    a31 = 0
    a32 = 0.5
    a41 = 0
    a42 = 0
    a43 = 1
    b1 = 1/6
    b2 = 1/3
    b3 = 1/3
    b4 = 1/6
    
    f1 = f(xk,u)
    f2 = f(xk + Δt*a21*f1,u)
    f3 = f(xk + Δt*a31*f1 + Δt*a32*f2,u)
    f4 = f(xk + Δt*a41*f1 + Δt*a42*f2 + Δt*a43*f3,u)

    xn = xk + Δt*(b1*f1 + b2*f2 + b3*f3 + b4*f4)
    
    return xn
end

function continuous_dynamics_differentiable(x, u, mode)
    dq = x[4:6]
    ddq, _ = solveEOM(x, mode, u)
    return [dq;ddq]
end

function discrete_dynamics(x, u, mode)
    xn = rk4_step((_x,_u)->continuous_dynamics_differentiable(_x,_u,mode),x,u)
    return xn
end

### (3) Hybrid system Description

# domain
# ineqs > 0; eqs = 0
function domain(x, contactMode)

    q = x[1:3]
    dq = x[4:6]
    
    # a(q) = or > 0
    
    a = compute_a(q)
    
    eqs_a = a[contactMode]
    ineqs_a = a[.~contactMode]
    
    # A_eq*dq = 0; A_geq*dq > 0
    
    A, _ = contact_mode_constraints(x, contactMode)
    
    eqs_A = A*dq
    
    ineqs = [zeros(0); ineqs_a]
    eqs = [zeros(0); eqs_a; eqs_A]
    
    return ineqs, eqs
end

function guard_set(x, mode_from, mode_to)
    
    ineqs = zeros(0)
    eqs = zeros(0)
    
    q = x[1:3]
    dq = x[4:6]
    
    d_ineqs, d_eqs = domain(x, mode_from)
    # c: d_ineqs > 0, d_eqs = 0
    ineqs = [ineqs; d_ineqs]
    eqs = [eqs; d_eqs]
    
    cs_mode_to = mode_to.==true
    cs_mode_from = mode_from.==true
    
    A_new, dA_new = contact_mode_constraints(x, mode_to)
    
    # if there is new impact
    new_cs = (cs_mode_from.==false).& (cs_mode_to.==true)
    if sum(new_cs) > 0
        a_new = compute_a(x)
        # c: a_new[new_cs] .== 0
        eqs = [eqs; a_new[new_cs]]
        
        A_impact, _ = contact_mode_constraints(x, new_cs)
        #c: A_impact*dq < 0
        ineqs = [ineqs; -A_impact*dq]
        
        p_hat = pinv(A_new*inv(M)*A_new')*A_new*dq
        #c: contact_force_constraints(p_hat, mode_to) > 0
        ineqs = [ineqs; -p_hat]
    end
        
    return ineqs, eqs
end

function jumpmap(x, mode_from, mode_to)

    dq_p, _ = computeResetMap(x, mode_to)
    
    return [x[1:3]; dq_p]
end

### (4) Animation
function rodshape(q)
    p1 = q[1:2] + R_2D(q[3])*[-l/2;0]
    p2 = q[1:2] + R_2D(q[3])*[l/2;0]
    pp = [p1 p2]
    return pp[1,:], pp[2,:]
end

function animation(x, y, θ, n, fps = 30)
    anim = Plots.Animation()
    for i ∈ 1:n
        px, py = rodshape([x[i],y[i],θ[i]])
        p = plot([-5,0,0],[0,0,-5], lw = 2, c=:black, xlims=(-1,1), ylims=(-1,1))
        plot!(p, px, py, lw = 3, aspect_ratio=:equal, c=:gray, opacity=.8, legend=false)
        frame(anim, p)
    end
    Plots.gif(anim, "anim.gif", fps = fps)
end


end