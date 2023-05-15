using OrthogonalLDA
using LinearAlgebra
using MultivariateStats
using Statistics

using Test


@testset "Basic" begin


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
        l = rand(1:3)
        label[i] = l
        X1[:,i] .= μ1[:,l] .+ 0.125*randn(d)
        X2[:,i] .= μ2[:,l] .+ 0.125*randn(d)
    end
    mstats1 = MultivariateStats.multiclass_lda_stats(X1, label)
    mstats2 = MultivariateStats.multiclass_lda_stats(X2, label)
    # concatenate the matrices
    Sw = fill(0.0, 2d,2d)
    Sb = fill(0.0, 2d,2d)

    Sw[1:d, 1:d] = mstats1.Sw
    Sb[1:d, 1:d] = mstats1.Sb

    Sw[d+1:2d, d+1:2d] = mstats2.Sw
    Sb[d+1:2d, d+1:2d] = mstats2.Sb

    u1,s1,v1 = svd(mstats1.Sb)
    u2,s2,v2 = svd(mstats2.Sb)
    
    # verify that the two between-scatter matrices are close to orthogonal
    #@show norm(u1[:,1:2]'*u2[:,1:2])

    # w, f0, f1 = OrthogonalLDA.orthogonal_lda(Sb, Sw, 2;debug=missing)
    w = OrthogonalLDA.orthogonal_lda(mstats1.Sb, mstats2.Sb, mstats1.Sw, mstats2.Sw, r;debug=missing)
	w1 = w[:, 1:r]
	w2 = w[:,r+1:2r]
	nn = norm(w1'*w2)
	@test nn < 1e-14
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

    @time w, f0, f1 = OrthogonalLDA.orthogonal_lda(Sb, Sw, 2;debug=missing)
  #μ1pp = w[1:d,:]*μ1p
  #μ2pp = w[d+1:2d,:]*μ2p

#    w = OrthogonalLDA.orthogonal_lda(mstats1.Sb, mstats2.Sb, mstats1.Sw, mstats2.Sw, 2)
#    Y1 = w[1:d, :]'*X1
#    Y2 = w[d+1:2d, :]'*X2
#    μ1p[:,1] = mean(Y1[:,label.==1],dims=2)
#    μ1p[:,2] = mean(Y1[:,label.==2],dims=2)
#    μ1p[:,3] = mean(Y1[:,label.==3],dims=2)
#
#    μ2p[:,1] = mean(Y2[:,label.==1],dims=2)
#    μ2p[:,2] = mean(Y2[:,label.==2],dims=2)
#    μ2p[:,3] = mean(Y2[:,label.==3],dims=2)
#    μ1pp = w[1:d,:]*μ1p
#    μ2pp = w[d+1:2d,:]*μ2p
#    @show μ1pp, μ1
#    @show μ2pp, μ2
end
