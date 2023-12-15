module DiffEqToleranceTools
using ForwardDiff
using DiffEqBase: norm

const errors = Tuple{Float64, Vector{Float64}, Vector{Float64}}[]

#norm(x) = sqrt(sum(x.^2)/length(x))

# function error_tracker(f)
#     function (u,p,t)
#         du = f(u,p,t)
#         if !any(isa.(u, ForwardDiff.Dual))
#             push!(errors, (t, u, du) )
#         end
#         return du
#     end
# end

function error_tracker(f)
    function (du, u, p, t)
        f(du,u,p,t)
        if t == 0.0
            empty!(errors)
        elseif !any(isa.(u, ForwardDiff.Dual)) 
            push!(errors, (t, copy(u), copy(du)) )
        end
    end
end

function iteration_time()
    es = unique(errors)
    ts = [x[1] for x in es]
    xs = [x[2] for x in es]
    errs = [x[3] for x in es]

    uts = unique(ts)
    
    return uts
end

function iteration_number()
    es = unique(errors)
    ts = [x[1] for x in es]
    xs = [x[2] for x in es]
    errs = [x[3] for x in es]

    uts = unique(ts)
    n = length(uts)

    ns = zeros(Int, n)
    for (i,t) in enumerate(uts)
        ns[i] = length(filter(x->x == t, ts))
    end

    return ns
end

function iteration_error(sol)
    
    es = deepcopy(unique(errors))
    ts = [x[1] for x in es]
    xs = [x[2] for x in es]
    
    prob = sol.prob
    u0 = prob.u0
    mass_matrix = prob.f.mass_matrix
    
    for e in es
        for i=1:length(u0)
            if mass_matrix[i,i] == 1
                e[3][i] = e[3][i] - ForwardDiff.derivative(t->sol(t; idxs=i), e[1])
            end
        end
    end
    errs = [norm(x[3]) for x in es]

    uts = unique(ts)
    n = length(uts)

    ferrs = zeros(n)
    for (i,t) in enumerate(uts)
        j = findlast(t .== ts)
        ferrs[i] = errs[j]
    end

    return ferrs
end

end # module DiffEqToleranceTools
