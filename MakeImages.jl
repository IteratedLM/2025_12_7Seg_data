#!/usr/bin/env julia

using Images, FileIO
using ImageTransformations: imrotate
using Interpolations: Linear
using Distributions, Random
using HDF5

# -----------------------------------------------------------------------------
# 0) Build a 7‑segment “perfect” 28×28 digit
# -----------------------------------------------------------------------------
function make_7seg(d::Integer; size::Int=28, margin::Int=2, thickness::Int=4)
    h = Int((size - 2*margin - 3*thickness) ÷ 2)
    s,m,t = size, margin, thickness
    segs = Dict(
      'A'=>((m+1):(m+t),           (m+t+1):(s-m-t)),
      'F'=>((m+t+1):(m+t+h),       (m+1):(m+t)),
      'B'=>((m+t+1):(m+t+h),       (s-m-t+1):(s-m)),
      'G'=>((m+t+h+1):(m+2*t+h),   (m+t+1):(s-m-t)),
      'E'=>((m+2*t+h+1):(m+2*t+2*h),(m+1):(m+t)),
      'C'=>((m+2*t+h+1):(m+2*t+2*h),(s-m-t+1):(s-m)),
      'D'=>((m+2*t+2*h+1):(m+3*t+2*h),(m+t+1):(s-m-t))
    )
    segNames=['A','B','C','D','E','F','G']

    glyph_map = Dict{Int, Vector{Char}}()

    for i in 1:128
        mask = i-1
        v = [ segNames[j] for j in 1:7 if (mask >> (j-1)) & 0x1 == 1 ]
        glyph_map[i] = v
    end
        
    oldkeys = collect(keys(glyph_map))
    digit_map = Dict{Int, Vector{Char}}(
        i => glyph_map[k] for (i,k) in enumerate(oldkeys)
            )

    img = zeros(Float32, size, size)
    for seg in digit_map[d]
      rs, cs = segs[seg]
      img[rs, cs] .= 1f0
    end
    return img
end

# -----------------------------------------------------------------------------
# 1) The noise+warp+downsample function
# -----------------------------------------------------------------------------
"""
    transform28(
      img28::Matrix{Float32},
      n::Int;
      mu::Float64=0.05,
      rho::Float64=0.05,
      shift_x::Int=0,
      shift_y::Int=0
    ) -> Matrix{Float32}

Pipeline:
  1. Upsample to (28n×28n) by pixel‑repeat.
  2. Dim all 1.0→max(0,1-ε) with ε∼Exp(1/mu).
  3. Rotate by θ∼N(0,rho) (subpixel, via `imrotate` + `Linear()`).
  4. Translate by (shift_x,shift_y) *small* pixels, zero‑fill.
  5. Downsample back to 28×28 by averaging each n×n block.
"""
function transform28(
    img28::Matrix{Float32},
    n::Int;
    mu::Float64=0.05,
    rho::Float64=0.05,
    sigma_x::Float64=0.0,
    sigma_y::Float64=0.0
)
    # (1) upsample
    big = repeat(img28, inner=(n,n))          # (28n × 28n)
    H, W = size(big)

    # (2) dimming
    ε = rand(Exponential(mu))
    b = max(0f0, 1f0 - ε)
    big .= ifelse.(big .== 1f0, b, 0f0)       # ones→b, zeros stay 0

    # (3) rotate about center
    θ = rand(Normal(0, rho))
    bigr = imrotate(big, θ;
        axes = axes(big),
        method = Linear(),
        fill = 0f0
    )

    # (4) integer‐pixel translate
    bigt = zeros(Float32, H, W)
    shift_x=round(Int,rand(Normal(0.0,sigma_x)))
    shift_y=round(Int,rand(Normal(0.0,sigma_y)))

    # positive shift_x moves right, shift_y moves down
    i1 = max(1,1+shift_y):min(H, H+shift_y)
    j1 = max(1,1+shift_x):min(W, W+shift_x)
    i0 = max(1,1-shift_y):min(H, H-shift_y)
    j0 = max(1,1-shift_x):min(W, W-shift_x)
    bigt[i1, j1] .= bigr[i0, j0]

    # (5) downsample by block‐average
    out = Array{Float32}(undef, 28, 28)
    @inbounds for i in 1:28, j in 1:28
        r0 = (i-1)*n + 1
        c0 = (j-1)*n + 1
        block = @view bigt[r0:r0+n-1, c0:c0+n-1]
        out[i,j] = mean(block)
    end

    return out
end

# -----------------------------------------------------------------------------
# 2) Example usage: save noisy 7‑segment digit PN­Gs
# -----------------------------------------------------------------------------

function save_noisy_digits(; n=5, mu=0.025, rho=0.02,
                            sigma_x=3.0, sigma_y=3.0, scale=10)
    for d in 0:9
        base = make_7seg(d)
        for k in 1:n
            img = transform28(base, n;
                mu=mu, rho=rho,
                sigma_x=sigma_x, sigma_y=sigma_y
            )

            # scale up for visibility
            imgbig = repeat(img, inner=(scale,scale))
            imgbig = map(clamp01nan, imgbig)
            save("noisy7seg_$(d)_$(k).png", Gray.(clamp.(imgbig,0f0,1f0)))
        end
    end
end

function save_digit(filename::String,img, scale=10)

    imgbig = repeat(img, inner=(scale,scale))
    imgbig = map(clamp01nan, imgbig)
    save(filename, Gray.(clamp.(imgbig,0f0,1f0)))

end



#––– 1) Generate and save the dataset –––#
"""
    save_noisy7seg_h5(path::String;
        n_per_digit::Int=500,
        mu::Float64=0.05,
        sigma::Float64=0.5,
        rho::Float64=0.05,
        Nsub::Int=4
    )

Generates `n_per_digit` noisy variants of each digit 0–9 via `noisy_7seg`, 
and writes them to `path` as an HDF5 file with datasets:
  - `"/imgs"`   of shape (28, 28, 10 * n_per_digit), dtype Float32  
  - `"/labels"` of length 10 * n_per_digit, dtype Int
"""
function save_noisy7seg_h5(path::String;
                           n_per_digit::Int=500,
    mu::Float64=0.025,
    sigma::Float64=6.0,
    rho::Float64=0.05,
                           nSub::Int=10
                           
                           )
    totalGlyphs=128
    total = totalGlyphs * n_per_digit
    imgs   = Array{Float32}(undef, 28, 28, total)
    base_imgs   = Array{Float32}(undef, 28, 28, totalGlyphs)
    labels = Vector{Int}(undef, total)
    base_labels = Vector{Int}(undef, totalGlyphs)
    idx = 1

    for d in 1:totalGlyphs
        base = make_7seg(d)
        base_imgs[:,:,d]=base
        base_labels[d]=d
        for i in 1:n_per_digit
            img = transform28(base, nSub;
                mu=mu, rho=rho,
                sigma_x=sigma, sigma_y=sigma
            )
            img = map(clamp01nan, img)
            imgs[:, :, idx]   = img
            labels[idx]       = d
            idx += 1
        end
    end

    h5open(path*".h5", "w") do f
        write(f, "imgs",   imgs)
        write(f, "labels", labels)
    end
    
    h5open(path*"_base.h5", "w") do f
        write(f, "imgs",   base_imgs)
        write(f, "labels", base_labels)
    end
    
    println("Saved dataset with $(total) images to `$path`.")
end

function load_noisy7seg_h5(path::String)
    h5open(path, "r") do f
        imgs   = read(f, "imgs")
        labels = read(f, "labels")
        return (imgs=imgs, labels=labels)
    end
end

#mu dimming
#sigma translation
#rho rotation


filename="digits_noise1"
save_noisy7seg_h5(filename,n_per_digit=100,mu=0.03,sigma=4.0,rho=0.05,nSub=10)


#filename="digits_noise2"
#save_noisy7seg_h5(filename,n_per_digit=100,mu=0.1,sigma=5.0,rho=0.1,nSub=10)


#filename="digits_noise3"
#save_noisy7seg_h5(filename,n_per_digit=100,mu=0.03,sigma=8.0,rho=0.25,nSub=10)
