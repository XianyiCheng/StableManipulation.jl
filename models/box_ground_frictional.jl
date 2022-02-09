# This file provides the model of 
# a 2D box on a flat ground at y = 0 with 2 frictional contacts at its bottom
# It includes: system properties, hybrid system description, functions for the ODE solver to simulate and animate the system


module BoxWorld 

using ForwardDiff
using Plots
using LinearAlgebra

### (1) system properties & contact constraints

const g = 9.81

const m = 1
const μ = 0.5

const w = 1.0 # width of the box
const h = 0.8 # height of the box
const M = [m 0 0; 0 m 0; 0 0 m*((h/2)^2+(w/2)^2)/3]

# set of contact modes
const modes = [0 0 0 0; # both free
        1 0 0 0; # sticking
        1 0 -1 0; # left-slide
        1 0 1 0; # right-slide
        0 1 0 0;
        0 1 0 -1;
        0 1 0 1;
        1 1 0 0;
        1 1 -1 -1;
        1 1 1 1] # both right-slide

const n_contacts = 2

# simulation step size
const Δt = 0.05
# tolerance for contacts
const tol_c = 1e-5

# 2D rotation matrix
function R_2D(θ)
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    return R
end

# constraints/contacts

# signed distance of contacts
function compute_a(q)
    y = q[2]
    θ = q[3]
    a1 = y - 0.5*h*cos(θ) - 0.5*w*sin(θ)
    a2 = y - 0.5*h*cos(θ) + 0.5*w*sin(θ)
    return [a1; a2]
end

# signed distance of contact in the contact tangent direction
function compute_a_t(q)
    p1 = R_2D(q[3])*[-w/2; -h/2] 
    p2 = R_2D(q[3])*[w/2; -h/2] 
    at1 = p1[1] + q[1]
    at2 = p2[1] + q[1]
    return [at1; at2]
end

# contact jacobian
function compute_A(q)
    θ = q[3]
    A = [0 1 0.5*h*sin(θ)-0.5*w*cos(θ); 0 1 0.5*h*sin(θ)+0.5*w*cos(θ)]
    return A
end

# the derivatives of contact jacobian
function compute_dA(q, dq)
    dA = reshape(ForwardDiff.jacobian(compute_A, q)*dq, n_contacts, 3)
    return dA
end

# contact tangent jacobian
function compute_A_tangent(q)
    A_t = ForwardDiff.jacobian(compute_a_t,q)
    return A_t
end

# the derivatives of contact tangent jacobian
function compute_dA_tangent(q, dq)
    dA_t = reshape(ForwardDiff.jacobian(compute_A_tangent, q)*dq, n_contacts, 3)
    return dA_t
end

function contact_mode_constraints(x, contactMode)
    # contact jacobian for equations of motion, only equalities
    q = x[1:3]
    dq = x[4:6]
    
    cs_mode = contactMode[1:n_contacts] .== 1
    ss_mode = contactMode[n_contacts+1:end]
    
    A = compute_A(q)
    A_t = compute_A_tangent(q)
    dA = compute_dA(q, dq)
    dA_t = compute_dA_tangent(q,dq)
    A = A[cs_mode,:]
    A_t = A_t[cs_mode,:]
    dA = dA[cs_mode,:]
    dA_t = dA_t[cs_mode,:]
    
    ss_active = ss_mode[cs_mode]
    
    A_all = zeros(0,3)
    A_all_f = zeros(0,3)
    dA_all = zeros(0,3)

    for k = 1:length(ss_active)
        ss = ss_active[k]
        if ss == 0
            A_all_f = [A_all_f; A[k,:]'; A_t[k,:]']
            dA_all = [dA_all; dA[k,:]'; dA_t[k,:]']
            A_all = [A_all; A[k,:]'; A_t[k,:]']
        else
            A_all_f = [A_all_f; A[k,:]'-ss*μ*A_t[k,:]']
            dA_all = [dA_all; dA[k,:]']
            A_all = [A_all; A[k,:]']
        end
    end
    
    return A_all_f, A_all, dA_all
end

function contact_force_constraints(λ, contactMode)
    # c > 0
    cs_mode = contactMode[1:n_contacts] .== 1
    ss_mode = contactMode[n_contacts+1:end]
    
    ss_active = ss_mode[cs_mode]
    
    i = 1
    ic = 1
    c = zeros(sum(ss_active.==0)*3 + sum(ss_active.!=0))
    for k = 1:length(ss_active)
        ss = ss_active[k]
        if ss == 0
            c[ic:ic+2] = [-λ[i]; -μ*λ[i] - λ[i+1]; -μ*λ[i] + λ[i+1]]
            ic += 3
            i += 2
        else
            c[ic] = -λ[i]
            i += 1
            ic += 1
        end
    end
    return c
end

function contact_velocity_constraints(x, contactMode)
    # contact constraints on velocities (cones of velocities), including equalities and inequalities 
    # reture A_eq, A_geq, A_eq*dq = 0, A_geq*dq >= 0
    q = x[1:3]
    dq = x[4:6]
    
    cs_mode = contactMode[1:n_contacts] .== 1
    ss_mode = contactMode[n_contacts+1:end]
    
    A = compute_A(q)
    A_t = compute_A_tangent(q)
    dA = compute_dA(q, dq)
    dA_t = compute_dA_tangent(q,dq)
    A = A[cs_mode,:]
    A_t = A_t[cs_mode,:]
    dA = dA[cs_mode,:]
    dA_t = dA_t[cs_mode,:]
    
    ss_active = ss_mode[cs_mode]
    
    AA_eq = zeros(0,3)
    AA_geq = zeros(0,3)
    dAA_eq = zeros(0,3)
    dAA_geq = zeros(0,3)

    for k = 1:length(ss_active)
        ss = ss_active[k]
        if ss == 0
            AA_eq = [AA_eq; A[k,:]'; A_t[k,:]']
            dAA_eq = [dAA_eq; dA[k,:]'; dA_t[k,:]']
        else
            AA_eq = [AA_eq; A[k,:]']
            dAA_eq = [dAA_eq; dA[k,:]']
            AA_geq = [AA_geq; ss*A_t[k,:]']
            dAA_geq = [dAA_geq; ss*dA_t[k,:]']
        end
    end
    
    return AA_eq, AA_geq, dAA_eq, dAA_geq
end

function contact_tangent_velocity_constraints(x, contactMode)
    # only return the constraints from contact tangent directions 
    # reture A_eq, A_geq, A_eq*dq = 0, A_geq*dq >= 0
    q = x[1:3]
    dq = x[4:6]
    
    cs_mode = contactMode[1:n_contacts] .== 1
    ss_mode = contactMode[n_contacts+1:end]
    
    A = compute_A(q)
    A_t = compute_A_tangent(q)
    dA = compute_dA(q, dq)
    dA_t = compute_dA_tangent(q,dq)
    A = A[cs_mode,:]
    A_t = A_t[cs_mode,:]
    dA = dA[cs_mode,:]
    dA_t = dA_t[cs_mode,:]
    
    ss_active = ss_mode[cs_mode]
    
    AA_eq = zeros(0,3)
    AA_geq = zeros(0,3)
    dAA_eq = zeros(0,3)
    dAA_geq = zeros(0,3)

    for k = 1:length(ss_active)
        ss = ss_active[k]
        if ss == 0
            AA_eq = [AA_eq; A_t[k,:]']
            dAA_eq = [dAA_eq; dA_t[k,:]']
        else
            AA_geq = [AA_geq; ss*A_t[k,:]']
            dAA_geq = [dAA_geq; ss*dA_t[k,:]']
        end
    end
    
    return AA_eq, AA_geq, dAA_eq, dAA_geq
end

### (2) function for simulation
### (2.1) functions for event-based simulation

function solveEOM(x, contactMode, u_control)
    # contactMode: bool vector, indicates which constraints are active
    q = x[1:3]
    dq = x[4:6]
    
    A_f, A, dA = contact_mode_constraints(x, contactMode)
    
    # compute EOM matrices
    N = [0; g; 0]
    C = zeros(3,3)
    Y = zeros(3)
    Y .= u_control
    #
    blockMat = [M A_f'; A zeros(size(A,1),size(A_f',2))] 

    b = [Y-N-C*dq; -dA*dq]
    
    #z = blockMat\b
    #println(blockMat)
    if rank(blockMat) < length(b)
#         z =pinv(blockMat)*b
        H = [zeros(3,size(blockMat,2)); zeros(size(blockMat,1)-3,3) 1e-6*I]
        z =(blockMat.+H)\b
    else
        z = blockMat\b
    end
    
    ddq = z[1:3]
    if (sum(contactMode[1:n_contacts])>=1)
        λ = z[4:end]
    else
        λ = []
    end
    
    return ddq, λ
end

function computeResetMap(x, contactMode)
    q = x[1:3]
    dq = x[4:6]

    A_f, A, dA = contact_mode_constraints(x, contactMode)
    
    c = size(A, 1)
    #
    blockMat = [M A_f'; A zeros(size(A,1),size(A_f',2))] 
    
    if rank(blockMat) < 3+c
        z = pinv(blockMat)*[M*dq; zeros(c)]
        p_hat = z[4:end]
        dq_p = z[1:3]
#         dq_p = z[1:3] .+ A\(-A*z[1:3])
    else
        z = blockMat\[M*dq; zeros(c)]
        dq_p = z[1:3]
        p_hat = z[4:end]
    end
    
    return dq_p, p_hat
end

function compute_FA(x, u_control)
    
    q = x[1:3]
    dq = x[4:6]
    
    a = compute_a(q)
    
    active_cs = abs.(a) .< tol_c
    inactive_cs = abs.(a) .> tol_c
    
    if sum(active_cs) == 0
        return zeros(4)
    end
    
    possibleModes = zeros(Int, 0, size(modes,2))
    
    contactMode = zeros(n_contacts*2)
    
    for k = 1:size(modes,1)
        m_cs = modes[k, 1:n_contacts].==1
        if length(findall(z->z==true, m_cs[inactive_cs])) == 0
            possibleModes = [possibleModes; modes[k, :]']
        end
    end

    max_cons = 0
    
    
    for kk = 1:size(possibleModes, 1)      
        
        m = possibleModes[kk,:]
        
        separate_cs = (m[1:n_contacts].!=1) .& active_cs
        _, A_separate, dA_separate = contact_mode_constraints(x, [separate_cs; ones(n_contacts)]) 

        ddq, λ = solveEOM(x, m, u_control)
        
        c_λ = contact_force_constraints(λ, m)
        
        if all(c_λ.>=0)
        
            As_eq, As_geq, dAs_eq, dAs_geq = contact_velocity_constraints(x, m)

            sep_vel_cond = ((A_separate*dq).>0) .| ((A_separate*ddq .+ dA_separate*dq).>=0)
            maintain_vel_cond = all(abs.(As_eq*dq).<tol_c) & all((As_geq*dq).>tol_c)

            if ~maintain_vel_cond
                if any((As_geq*dq).<tol_c)
                    maintain_vel_cond = all(abs.(dAs_eq*dq + As_eq*ddq).<tol_c) & all((dAs_geq*dq + As_geq*ddq).>0)
                end
            end

            if all(c_λ.>=0) && all(sep_vel_cond) && maintain_vel_cond
    
                if sum(m[1:n_contacts]) > max_cons
                    contactMode = m
                    max_cons = sum(m[1:n_contacts])
                end
            end
        end
    end
    
    return contactMode
end

function compute_IV(x)
    
    q = x[1:3]
    dq = x[4:6]
    
    a = compute_a(q)
    
    active_cs = abs.(a) .< tol_c
    inactive_cs = abs.(a) .> tol_c
    
    if sum(active_cs) == 0
        return zeros(Int, 4)
    end
    
    possibleModes = zeros(Bool, 0, size(modes,2))
    
    contactMode = zeros(n_contacts*2)
    
    for k = 1:size(modes,1)
        m_cs = modes[k, 1:n_contacts].==1
        if length(findall(z->z==true, m_cs[inactive_cs])) == 0
            possibleModes = [possibleModes; modes[k, :]']
        end
    end
    
    max_cons = 0
    
    for kk = 1:size(possibleModes, 1)
        
        m = possibleModes[kk,:]
        
        separate_cs = (m[1:n_contacts].!=1) .& active_cs
        _, A_separate, _ = contact_mode_constraints(x, [separate_cs; ones(n_contacts)]) 

        dq_p, p_hat = computeResetMap(x, m)
        
        c_p_hat = contact_force_constraints(p_hat, m)
        
        As_eq, As_geq, _, _ = contact_velocity_constraints(x, m)
        
        if all(c_p_hat.>=0) && all(A_separate*dq_p.>=0) && all(abs.(As_eq*dq_p).<tol_c) && all(As_geq*dq_p.>0)
            if sum(m[1:n_contacts]) > max_cons
                contactMode = m
                max_cons = sum(m[1:n_contacts])
            end
        end
    end
    
    return contactMode
end

function mode(x, u_control)
    return Vector{Int}(compute_FA(x, u_control))
end

function guard_conditions(x, contactMode, u_control)
    q = x[1:3]
    dq = x[4:6]
    
    a = compute_a(q)
#     a[contactMode[1:n_contacts] .== 1] .= 0.0
    
    # for sliding->sticking transition
    v_all = zeros(n_contacts)
#     _, As_geq, _, _ = contact_velocity_constraints(x, contactMode)
#     v_all[1:size(As_geq,1)] = -As_geq*dq
    At_eq, At_geq, _, _ = contact_tangent_velocity_constraints(x, contactMode)
    v_all[1:size(At_geq,1)] = -At_geq*dq
    
    ddq, λ = solveEOM(x, contactMode, u_control)
    c_λ = contact_force_constraints(λ, contactMode)
    c_λ_all = zeros(3*n_contacts)
    c_λ_all[1:length(c_λ)] = c_λ
    
    c = [a; v_all; c_λ_all]
#     if debug == true     
# #         println("guard_conditions ", c)
#     end
    
    dir = [-ones(Int,length(a)); ones(Int,length(v_all)); ones(Int,length(c_λ_all))]
    
    return c, dir
end

### (2.2) wrapper for ode solver

function ode_dynamics!(dx, x, p, t)
    # p from integrator: (contact mode, controller, t_control, h_control, u_control)
    
    # if debug == true
    #     println("Dynamics evalutation ", t, "x ", x)
    # end
    
    # bound the simulation
    if any(abs.(x[1:3]).>10) || any(abs.(x).>50) || (x[2] < -0.2)
        dx .= [x[4:6]; zeros(3)]
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
        # if debug == true
        #     println("Control evaluation ", u_control)
        # end
    end
    
    ddq, λ = solveEOM(x, contactMode, u_control)
    
    #println("acceleration ", ddq)
    
    dx .= [dq; ddq]
    c_λ = contact_force_constraints(λ, contactMode)
    if(sum(c_λ.<-tol_c) > 0)
#         println(c_λ)
        new_contactMode = compute_FA(x, u_control)
        if (all(new_contactMode .==0))
            return
        end
        p[1] .= reshape(new_contactMode,size(p[1]))
        # if debug == true
        #     println("New mode from constraints ", new_contactMode)
        #     println(λ)
        # end
    end
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
        # if debug == true
        #     println("New mode from FA: ", new_contactMode)
        #     println(x, u_control)
        # end
    end
    # constraints
#     if dir[idx] < 0
#         new_contactMode = contactMode
#         new_contactMode[idx] = false
#     end
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
        if (all(new_contactMode .==0))
            return
        end
        controller = integrator.p[2]
        u_control = controller(x, new_contactMode)
        integrator.p[5] .= u_control
        
        dq_p, p_hat = computeResetMap(x, new_contactMode)
        integrator.u .= [x[1:3]; dq_p]
        integrator.p[1] .= reshape(new_contactMode,size(integrator.p[1]))
        
        # if debug == true
        #     println("New mode from IV: ", new_contactMode)
        #     println(x)
        # end

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
    q = x[1:3]
    dq = x[4:6]
    
    A_f, A, dA = contact_mode_constraints(x, mode)
    
    # contact constraints

    # compute EOM matrices
    N = [0; g; 0]
    C = zeros(3,3)
    # Y = zeros(3)
    Y = u
    
    blockMat = [M A_f'; A zeros(size(A,1),size(A_f',2))] 

    b = [Y-N-C*dq; -dA*dq]
    
    H = [zeros(3,size(blockMat,2)); zeros(size(blockMat,1)-3,3) 1e-6*I]
    z =(blockMat.+H)\b

    ddq = z[1:3]
    
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
    
    cs = contactMode[1:n_contacts] .== 1
    ss = contactMode[n_contacts+1:end]
    
    q = x[1:3]
    dq = x[4:6]
    
    # a(q) = or > 0
    
    a = compute_a(q)
    
    eqs_a = a[cs]
    ineqs_a = a[.~cs]
    
    # A_eq*dq = 0; A_geq*dq > 0
    
    A_eq, A_geq, _, _ = contact_velocity_constraints(x, contactMode)
    
    eqs_A = A_eq*dq
    ineqs_A = A_geq*dq
    
    # separating contacts
    separate_cs = (cs.!=1) .& (abs.(a) .< tol_c)
    _, A_separate, _ = contact_mode_constraints(x, [separate_cs; ones(n_contacts)])
    
    ineqs_A_sep = A_separate*dq
    
    ineqs = [zeros(0); ineqs_a; ineqs_A]#; ineqs_A_sep]
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
    
    cs_mode_to = mode_to[1:n_contacts].==1
    cs_mode_from = mode_from[1:n_contacts].==1
    
    A_f_new, A_new, dA_new = contact_mode_constraints(x, mode_to)
    
    # if there is new impact
    new_cs = (cs_mode_from.==false).& (cs_mode_to.==true)
    if sum(new_cs) > 0
            a_new = compute_a(x)
        # c: a_new[new_cs] .== 0
        eqs = [eqs; a_new[new_cs]]
        
        _, A_impact, _, = contact_mode_constraints(x, [new_cs; ones(n_contacts)])
        #c: A_impact*dq < 0
        ineqs = [ineqs; -A_impact*dq]
        
        p_hat = pinv(A_new*inv(M)*A_f_new')*A_new*dq
        #c: contact_force_constraints(p_hat, mode_to) > 0
        ineqs = [ineqs; contact_force_constraints(p_hat, mode_to)]
    end
    
    # if it is stick -> slide: do nothing, guard = domain
    # if it is slide -> stick: A_stick dq = 0
    ss_mode_to = mode_to[n_contacts+1:end]
    ss_mode_from = mode_from[n_contacts+1:end]
    ss_new_stick = (cs_mode_from.==true).& (cs_mode_to.==true) .& (ss_mode_to .== 0) .& (ss_mode_from .!=0)
    
    if sum(ss_new_stick) > 0    
        A_stick, _, _, _ = contact_velocity_constraints(x, [ss_new_stick; zeros(n_contacts)])
        # c: A_stick*dq = 0
        eqs = [eqs; A_stick*dq]
    end
        
    return ineqs, eqs
end

function jumpmap(x, mode_from, mode_to)
    
    q = x[1:3]
    dq = x[4:6]

    A_f, A, dA = contact_mode_constraints(x, mode_to)
    
    c = size(A, 1)
    #
    blockMat = [M A_f'; A zeros(size(A,1),size(A_f',2))] 
    
    b = [M*dq; zeros(c)]

    H = [zeros(3,size(blockMat,2)); zeros(size(blockMat,1)-3,3) 1e-6*I]
    z =(blockMat.+H)\b
        
    dq_p = z[1:3]
    p_hat = z[4:end]
    
    return [x[1:3]; dq_p]
end

### (4) Animation

function boxshape(q)
    p1 = q[1:2] + R_2D(q[3])*[w/2;h/2]
    p2 = q[1:2] + R_2D(q[3])*[w/2;-h/2]
    p3 = q[1:2] + R_2D(q[3])*[-w/2;-h/2]
    p4 = q[1:2] + R_2D(q[3])*[-w/2;h/2]
    pp = [p1 p2 p3 p4]
    return Shape(pp[1,:], pp[2,:])
end

function animation(x, y, θ, n, fps = 30)
    anim = Plots.Animation()
    for i ∈ 1:n
        p = plot([-10,5],[0,0], lw = 2, c=:black, xlims=(-3,3), ylims=(-0.5,3))
        plot!(p, boxshape([x[i],y[i],θ[i]]), aspect_ratio=:equal, c=:gray, opacity=.5, legend=false)
        frame(anim, p)
    end
    Plots.gif(anim, "anim.gif", fps = fps)
end

### end of the module Box
end 
