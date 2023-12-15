using Test
using DiffEqToleranceTools
using ModelingToolkit
using DAE2AE
using OrdinaryDiffEq
using CairoMakie

@parameters t
D = Differential(t)
vars = @variables x(t)=0

d=1
k=1000
Δt=1e-4
F = 100

tol = 1e-3

eqs =[
    D(x) ~ (F - k*x^1.5)/d
]

@named odesys = ODESystem(eqs, t, vars, [])
sys = structural_simplify(odesys)
f = ODEFunction(sys)
f′ = ODEFunction(DiffEqToleranceTools.error_tracker(f.f.f_iip); mass_matrix = f.mass_matrix)
prob = ODEProblem(f′, [0.0], (0, Δt*99), [])

sol1=solve(prob, ImplicitEuler(nlsolve=NLNewton(always_new=true, relax=0//10, max_iter=100)); abstol=tol, reltol=tol, initializealg=NoInit())
err1 = DiffEqToleranceTools.iteration_error(sol1)
n1 = DiffEqToleranceTools.iteration_number()

sol2=solve(prob, ImplicitEuler(nlsolve=NLNewton(always_new=false, relax=0//10, max_iter=100)); abstol=tol, reltol=tol, initializealg=NoInit())
err2 = DiffEqToleranceTools.iteration_error(sol2)
n2 = DiffEqToleranceTools.iteration_number()

sol3=solve(prob, ImplicitEuler(nlsolve=NLNewton(always_new=true, relax=4//10, max_iter=100)); abstol=tol, reltol=tol, initializealg=NoInit())
err3=DiffEqToleranceTools.iteration_error(sol3)
n3=DiffEqToleranceTools.iteration_number()

fig = Figure()
ax = Axis(fig[1,1], yscale=log10, ylabel="norm(du)")
lines!(ax, err1; label="NLNewton(always_new=true, relax=0//10)")
lines!(ax, err2; label="NLNewton(always_new=false, relax=0//10)")
lines!(ax, err3; label="NLNewton(always_new=true, relax=4//10)")
# lines!(ax, ones(9)*tol; label="tol", color=:black, linewidth=2)
Legend(fig[1,2], ax)
ylims!(ax, 1e-16, 1e-1)

ax = Axis(fig[2,1], ylabel="iterations", xlabel="step")
scatterlines!(ax, n1; label="NLNewton(always_new=true, relax=0//10)")
scatterlines!(ax, n2; label="NLNewton(always_new=false, relax=0//10)")
scatterlines!(ax, n3; label="NLNewton(always_new=true, relax=4//10)")
ylims!(ax, 0, 20)

fig

