using OrthogonalLDA
using LinearAlgebra
using MultivariateStats
using Statistics
using StableRNGs

using Test

@testset "PCA" begin
	rng = StableRNG(1234)
	q,r = qr(randn(rng, 4,4))
	W1 = q[:,1:2]
	W2 = q[:, 3:4]
	
	r1 = randn(rng, 2,2)
	Σ1 = W1*r1'*r1*W1' .+ 0.0*randn(rng, 4,4)

	r2 = randn(rng, 2,2)
	Σ2 = W2*r2'*r2*W2' + 0.0*randn(rng, 4,4)

	W = OrthogonalLDA.orthogonal_pca([Σ1, Σ2], [2,2];debug=missing,RNG=rng)
	W1p = W[:,1:2]
	W2p = W[:, 3:4]
	# test that the solutions overlap with the ground truth
	@show  norm(W1p - W1)
	u,s,v = svd(W1p'*W1)
    @test all(abs.(s .- 1.0) .< sqrt(eps(Float64)))
	u,s,v = svd(W2p'*W2)
    @test all(abs.(s .- 1.0) .< sqrt(eps(Float64)))

	# test that the solutions do not overlap across 
	u,s,v = svd(W2p'*W1)
    @test all(s .< sqrt(eps(Float64)))
	u,s,v = svd(W1p'*W2)
    @test all(s .< sqrt(eps(Float64)))
end

@testset "LDA" begin
	rng = StableRNG(1234)
    d = 4
	nc = 3
	r = nc-1
    # 1st plane
    μ1 = fill(0.0, d, 3)
    μ1[:,1] = [0.0,1.0, 0.0, 0.0]
    μ1[:,2] = [1.0, 0.0, 0.0, 0.0]
    μ1[:,3] = [1.0, 1.0, 0.0, 0.0]

    # 2nd plane
    μ2  = fill(0.0, d, 3)
    μ2[:,1] = [0.0, 0.0, 0.0, 1.0]
    μ2[:,2] = [0.0, 0.0, 1.0, 0.0]
    μ2[:,3] = [0.0, 0.0, 1.0, 1.0]


    nt = 1000
    label = fill(0,nt)
    X1 = fill(0.0, d ,nt)
    X2 = fill(0.0, d ,nt)
    for i in 1:nt
        l = rand(rng, 1:3)
        label[i] = l
        X1[:,i] .= μ1[:,l] .+ 0.125*randn(rng, d)
        X2[:,i] .= μ2[:,l] .+ 0.125*randn(rng, d)
    end
    mstats1 = MultivariateStats.multiclass_lda_stats(nc, X1, label)
    mstats2 = MultivariateStats.multiclass_lda_stats(nc, X2, label)
    # concatenate the matrices
    Sw = fill(0.0, 2d,2d)
    Sb = fill(0.0, 2d,2d)

    Sw[1:d, 1:d] = mstats1.Sw
    Sb[1:d, 1:d] = mstats1.Sb

    Sw[d+1:2d, d+1:2d] = mstats2.Sw
    Sb[d+1:2d, d+1:2d] = mstats2.Sb

    u1,s1,v1 = svd(mstats1.Sb)
    u2,s2,v2 = svd(mstats2.Sb)

	wr = OrthogonalLDA.orthogonal_lda([mstats1.Sb, mstats2.Sb], [mstats1.Sw, mstats2.Sw], [r,r];debug=missing, RNG=rng)
	w1 = wr[:, 1:r]
	w2 = wr[:,r+1:2r]
	nn = norm(w1'*w2)
    @test nn < sqrt(eps(Float64))
    Y1 = w1'*X1
    Y2 = w2'*X2
    μ1p = fill(0.0, 2,3)
    μ1p[:,1] = mean(Y1[:,label.==1],dims=2)
    μ1p[:,2] = mean(Y1[:,label.==2],dims=2)
    μ1p[:,3] = mean(Y1[:,label.==3],dims=2)

    μ2p = fill(0.0, 2,3)
    μ2p[:,1] = mean(Y2[:,label.==1],dims=2)
    μ2p[:,2] = mean(Y2[:,label.==2],dims=2)
    μ2p[:,3] = mean(Y2[:,label.==3],dims=2)

    #decode
    p1 = 0.0
    p2 = 0.0
    for i in 1:nt
        dd = dropdims(sum(abs2, Y1[:,i] .- μ1p,dims=1),dims=1)
        p1 += (argmin(dd) == label[i])
        dd = dropdims(sum(abs2, Y2[:,i] .- μ2p,dims=1),dims=1)
        p2 += (argmin(dd) == label[i])
    end
    p1 /= nt
    p2 /= nt
    # make sure we can decode the responses
    @test p1 > 0.9
    @test p2 > 0.9

end
