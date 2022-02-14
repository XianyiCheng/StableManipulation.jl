function R_2D(θ)
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    return R
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