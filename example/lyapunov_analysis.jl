using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Plots
using Convex, SCS
using JLD

A = [-1 0.5; -3 -1]
# A = [-1 0; 0 -1]

solver = () -> SCS.Optimizer(verbose=true)

P = Semidefinite(2)

prob1 = minimize(tr(P))
prob1.constraints += isposdef( -A'*P - P*A - 0.01*Matrix(I,2,2))

# Convex.solve!(prob1,solver)

# println(evaluate(P))

# 
B = [0.1 0; 0 0.1]

Q = Semidefinite(2)
Y = Variable((2,2))

prob = maximize(tr(Q))
prob.constraints += isposdef( -(A*Q + Q*A' + B*Y + (B*Y)') - 1e-8*Matrix(I,2,2))
Convex.solve!(prob,solver)

Q_sol = evaluate(Q)
Y_sol = evaluate(Y)

println(IOContext(stdout, :compact => true),"Q\n", Q_sol)
println(IOContext(stdout, :compact => true), "Y\n",Y_sol)

P = inv(0.5*(Q_sol + Q_sol'))
K = Y_sol*P

println(isposdef(P))
JV = -(A*Q_sol + Q_sol*A' + B*Y_sol + (B*Y_sol)')
println(isposdef(JV + JV'))

# results: it can find a Lyapunov function P and linear state feedback K for a linearized system