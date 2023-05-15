module OrthogonalLDA

using Manifolds
using Manopt
using ManifoldsBase
using LinearAlgebra

import Manifolds: _vector_transport_to

struct IdentityTransport <: AbstractVectorTransportMethod
end

Manifolds._vector_transport_to(M::Stiefel, Y, p, X, q, T::IdentityTransport) = (Y .= project(M, q, X))

function orthogonal_lda(Sb::AbstractMatrix{Float64}, Sw::AbstractMatrix{Float64}, r::Int64;debug::Union{Missing, Symbol}=:default, method=:quasi_newton)
    if !ismissing(debug) && debug == :default
		debug=[:Iteration, " ", :Cost, "\n", 1, :Stop]
	end
	d = size(Sb,1)
	d == size(Sb,2) == size(Sw,1) == size(Sw,2) || error("Matrices must be of equal size")
	M = Stiefel(d, r)
	
	function F(::Stiefel, W::Array{Float64,2})
		-tr(W'*Sb*W)/tr(W'*Sw*W)
	end
	
	#G = fill(0.0, d, r)
	function gradF(M, W::Array{Float64,2})
		trb = tr(W'*Sb*W)
		trw = tr(W'*Sw*W)
		G = -((Sb*W + Sb'*W).*trw - trb.*(Sw*W + Sw'*W))./trw^2
		return project(M, W, G)
	end
	w = rand(M)
	f0 = F(M, w)
	if method == :quasi_newton
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
	elseif method == :gradient_descent
		w = gradient_descent(
		    M,
		    F,
		    gradF,
		    w;
		    stopping_criterion=StopWhenGradientNormLess(norm(M, w, gradF(M, w)) * 10^(-8)),
		    debug=debug,
		)
	elseif method == :trust_regions
		w = trust_regions(M, 
						  F, 
						  gradF, 
						  ApproxHessianFiniteDifference(M, w, gradF),
						  w;
						#vector_transport_method=IdentityTransport(),
		    			#stopping_criterion=StopWhenGradientNormLess(norm(M, w, gradF(M, w)) * 10^(-8)),
		    			debug=debug
		)
	else
		error("Unknown method $(method)")
	end
	f1 = F(M, w)
	w, f0, f1
end

function orthogonal_lda(Sb1::AbstractMatrix{Float64}, Sb2::AbstractMatrix{Float64},Sw1::AbstractMatrix{Float64}, Sw2::AbstractMatrix{Float64}, r::Int64;debug=:default, method=:quasi_newton)
    d1 = size(Sb1,1)
    d2 = size(Sb2,1)

	if !ismissing(debug) && debug == :default
		debug=[:Iteration, " ", :Cost, "\n", 1, :Stop]
	end

	function F(::Stiefel, W::Array{Float64,2})
        W1 = W[1:d1,:]
        W2 = W[d1+1:d1+d2,:]
        a = -tr(W1'*Sb1*W1)/tr(W1'*Sw1*W1)
        b = -tr(W2'*Sb2*W2)/tr(W2'*Sw2*W2)
        a + b
    end

	function gradF(M, W::Array{Float64,2})
        W1 = W[1:d1,:]
        W2 = W[d1+1:d1+d2,:]
		trb1 = tr(W1'*Sb1*W1)
		trb2 = tr(W2'*Sb2*W2)
		trw1 = tr(W1'*Sw1*W1)
		trw2 = tr(W2'*Sw2*W2)
		G1 = -((Sb1*W1 + Sb1'*W1).*trw1 - trb1.*(Sw1*W1 + Sw1'*W1))./trw1^2
		G2 = -((Sb2*W2 + Sb2'*W2).*trw2 - trb2.*(Sw2*W2 + Sw2'*W2))./trw2^2
        return project(M, W, cat(G1,G2,dims=1))
	end

	M = Stiefel(d1+d2, r)
	w = rand(M)
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
end # module
