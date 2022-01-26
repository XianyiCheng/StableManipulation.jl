
"""
riccati(A,B,Q,R,Qf,N)

Use backward riccati recursion to solve the finite-horizon time-invariant LQR problem.
Returns vectors of the feedback gains `K` and cost-to-go matrices `P`, where `length(K) == N-1`,
`length(P) == N`, and `size(K[i]) == (m,n)` and `size(P[i]) == (n,n)`.

# Arguments:
* `A`: `(n,n)` discrete dynamics Jacobian wrt the state
* `B`: `(n,m)` discrete dynamics Jacobian wrt the control
* `Q`: `(n,n)` stage-wise cost matrix for states
* `R`: `(m,m)` stage-wise cost matrix for controls
* `Qf`: `(n,n)` cost matrix for terminal state
* `N`: Integer number of time steps (horizon length).
"""
function riccati(A,B,Q,R,Qf,N)
    # initialize the output
    n,m = size(B)
    P = [zeros(n,n) for k = 1:N]
    K = [zeros(m,n) for k = 1:N-1]

    P[end] .= Qf
    for k = reverse(1:N-1) 
        K[k] .= (R + B'P[k+1]*B)\(B'P[k+1]*A)
        P[k] .= Q + A'P[k+1]*A - A'P[k+1]*B*K[k]
    end

    # return the feedback gains and ctg matrices
    return K,P
end