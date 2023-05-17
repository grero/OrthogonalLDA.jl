module OrthogonalLDA

using Manifolds
using Manopt
using ManifoldsBase
using LinearAlgebra
using Random

import Manifolds: _vector_transport_to

struct IdentityTransport <: AbstractVectorTransportMethod
end

Manifolds._vector_transport_to(M::Stiefel, Y, p, X, q, T::IdentityTransport) = (Y .= project(M, q, X))

function orthogonal_lda(Sb::Vector{T}, Sw::Vector{T}, r::Vector{Int64};debug=:default, method=:quasi_newton, RNG=Random.GLOBAL_RNG) where T <: AbstractMatrix{Float64}
	d = size(first(Sb),1)
	n = length(Sb)
	for sb in Sb
		if size(sb,1) != d
			error("All dimensions must be equal")
		end
	end

	if !ismissing(debug) && debug == :default
		debug=[:Iteration, " ", :Cost, "\n", 1, :Stop]
	end

	function F(::Stiefel, W::Array{Float64,2})
		a = 0.0
		offset = 0
		for (_Sb, _Sw,_r) in zip(Sb,Sw,r)
			_W = W[:,(offset+1):(offset+_r)]
			a += -tr(_W'*_Sb*_W)/tr(_W'*_Sw*_W)
			offset += _r
		end
		a
    end

	function gradF(M, W::Array{Float64,2})
		G = fill(0.0, d, sum(r))
		offset = 0
		for (_Sb, _Sw,_r) in zip(Sb, Sw,r)
			_W = W[:,(offset+1):(offset+_r)]
			trb = tr(_W'*_Sb*_W)
			trw = tr(_W'*_Sw*_W)
			G[:,(offset+1):(offset+_r)] = -((_Sb*_W + _Sb'*_W).*trw - trb.*(_Sw*_W + _Sw'*_W))./trw^2
			offset += _r
		end
        return project(M, W, G)
	end

	M = Stiefel(d, sum(r))
	w = rand(RNG, M)
    w = quasi_Newton(
        M,
        F,
        gradF,
        w;
        memory_size=32,
        #evaluation=MutatingEvaluation(),
        cautious_update=true,
        vector_transport_method=ProjectionTransport(),
        stopping_criterion=StopWhenGradientNormLess(norm(M, w, gradF(M, w)) * 10^(-9)),
        debug=debug,
    )
end

function orthogonal_pca(Σ::Vector{T}, r::Vector{Int64};debug=:default, method=:quasi_newton, RNG=Random.GLOBAL_RNG) where T <: AbstractMatrix{Float64}
	d = size(first(Σ),1)
	n = length(Σ)
	for S in Σ
		if size(S,1) != d
			error("All dimensions must be equal")
		end
	end
	S = fill(0.0, n)
	for (i,_Σ) in enumerate(Σ)
		u,s,v = svd(_Σ)
		S[i] = sum(s)
	end
	if !ismissing(debug) && debug == :default
		debug=[:Iteration, " ", :Cost, "\n", 1, :Stop]
	end

	function F(::Stiefel, W::Array{Float64,2})
		a = 0.0
		offset = 0
		for (_Σ, _r, _s) in zip(Σ,r,S)
			_W = W[:,(offset+1):(offset+_r)]
			a += tr(_W'*_Σ*_W)/_s
			offset += _r
		end
		-a/n
    end

	function gradF(M, W::Array{Float64,2})
		G = fill(0.0, d, sum(r))
		offset = 0
		for (_Σ, _s,_r) in zip(Σ,S, r)
			_W = W[:,(offset+1):(offset+_r)]
			G[:,(offset+1):(offset+_r)] = -(_Σ*_W + _Σ'*_W)./(_s*n)
			offset += _r
		end
        return project(M, W, G)
	end

	M = Stiefel(d, sum(r))
	w = rand(RNG, M)
    w = quasi_Newton(
        M,
        F,
        gradF,
        w;
        memory_size=32,
        #evaluation=MutatingEvaluation(),
        cautious_update=true,
        vector_transport_method=ProjectionTransport(),
        stopping_criterion=StopWhenGradientNormLess(norm(M, w, gradF(M, w)) * 10^(-14)),
        debug=debug,
    )
end
end # module
