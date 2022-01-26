using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Plots
using Convex, SCS
using JLD
using StableManipulation

# the model includes: TODO
include("../models/box_ground_frictional.jl")

function no_controller(x, contactMode)
    return zeros(3)
end

dynamics! = BoxWorld.ode_dynamics!
conditions = BoxWorld.ode_conditions
affect! = BoxWorld.ode_affect!
affect_neg! = BoxWorld.ode_affect_neg!

controller = no_controller

x0 = [0;1;0.3;0;0;0]
initial_mode = [0,0,0,0]

tspan = (0.0, 3.0)
callback_length = 5*BoxWorld.n_contacts

h_control = BoxWorld.Δt

prob = ODEProblem(dynamics!, x0, tspan, (initial_mode, controller, [0.0], h_control, zeros(3)))
cb = VectorContinuousCallback(conditions, affect!, affect_neg!, callback_length)
sol = solve(prob, Tsit5(); callback = cb, abstol=1e-15,reltol=1e-15, adaptive=false,dt=BoxWorld.Δt)
println("Simulation status: ", sol.retcode)

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