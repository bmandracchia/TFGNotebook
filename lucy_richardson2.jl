using DeconvOptim: gradient, get_plan, fft_or_rfft, p_conv_aux!

"""
    deconvRL(measured, psf; psf_bp=nothing, regularizer=nothing, λ=0.05, iterations=100, conv_dims=1:ndims(psf), threshold=0, progress=nothing)

    Classical iterative Richardson-Lucy iteration scheme for deconvolution.
    `measured` is the measured array and `psf` the point spread function.
    Converges slower than the optimization approach of `deconvolution`.

    # Keyword Arguments
    - `psf_bp=nothing`: A psf for backpropagation. If not provided, the conjugate of the OTF is used.
    - `regularizer=nothing`: A regularizer function. Can be used to impose constraints or smoothness.
    - `λ=0.05`: A float indicating the total weighting of the regularizer with respect to the global loss function.
    - `iterations=100`: Specifies the number of iterations to perform.
    - `conv_dims=1:ndims(psf)`: Specifies the dimensions along which convolution is performed.
    - `threshold=0`: Minimum threshold for the reconstructed image to avoid negative values.
    - `progress=nothing`: If not `nothing`, progress will be monitored and logged.

    # Example
"""
function deconvRL(
    measured::AbstractArray{T}, psf::AbstractArray{T};
    psf_bp::Union{Nothing, AbstractArray{T}} = nothing,
    regularizer::Union{Nothing, Function} = nothing,
    λ::Float64 = 0.05,
    iterations::Int = 100,
    conv_dims::AbstractVector{Int} = 1:ndims(psf),
    threshold::T = zero(T),
    progress::Union{Nothing, Function} = nothing
) where {T}

    otf, conv_temp = plan_conv(measured, psf, conv_dims)

    # Initialize otf_conj based on psf_bp
    otf_conj = isnothing(psf_bp) ? conj.(otf) : plan_conv(measured, psf_bp, conv_dims)[1]

    # Apply threshold and initialize reconstruction
    rec = max.(measured, threshold)

    # Precompute gradient buffer if regularizer is provided
    buffer_grad = isnothing(regularizer) ? nothing : similar(rec)
    if !isnothing(regularizer)
        buffer_grad .= Base.invokelatest(gradient, regularizer, rec)[1]
    end
    ∇reg(x) = buffer_grad .= Base.invokelatest(gradient, regularizer, x)[1]

    buffer = similar(measured)  # Preallocate once

    # Define iteration functions
    function iter!(rec)
        buffer .= measured ./ (conv_temp(rec, otf))
        buffer .= conv_temp(rec, otf_conj)
        if !isnothing(regularizer)
            rec .-= λ .* ∇reg(rec)
        end
    end

    # Loss function for logging
    loss = progress === nothing ? nothing : (myrec -> begin
        fwd = conv_temp(myrec, otf)
        sum(fwd .- measured .* log.(fwd))
    end)

    # Logging setup
    if progress !== nothing
        record_progress!(progress, rec, 0, loss(rec), 0.0, 1.0)
    end

    # Perform iterations
    for i in 1:iterations
        iter!(rec)
        if progress !== nothing
            record_progress!(progress, copy(rec), i, loss(rec), 0.0, 1.0)
        end
    end

    return rec
end

# The following code is commented out as it is not used in the current implementation.
# function deconvRL(measured::AbstractArray{T}, psf::AbstractArray{T}; 
#                   psf_bp::Union{Nothing, AbstractArray{T}}=nothing,
#                   regularizer::Union{Nothing, Function}=nothing,
#                   λ::Float64=0.05,
#                   iterations::Int=100,
#                   conv_dims::AbstractVector{Int}=1:ndims(psf),
#                   threshold::T=zero(T),
#                   progress::Union{Nothing, Function}=nothing) where {T}

#     otf, conv_temp! = plan_conv!(measured, psf, conv_dims)

#     # Initialize otf_conj based on psf_bp
#     otf_conj = isnothing(psf_bp) ? conj.(otf) : plan_conv(measured, psf_bp, conv_dims)[1]

#     # Apply threshold and initialize reconstruction
#     rec = max.(measured, threshold)

#     # Precompute gradient buffer if regularizer is provided
#     buffer_grad = isnothing(regularizer) ? nothing : similar(rec)
#     ∇reg = isnothing(regularizer) ? x -> nothing : x -> begin
#         Base.invokelatest(gradient, regularizer, x)[1]
#     end

#     buffer = similar(measured)  # Preallocate once

#     # Define iteration functions
#     function iter!(rec)
#         conv_temp!(buffer, rec, otf)
#         buffer .= measured ./ buffer
#         conv_temp!(buffer, buffer, otf_conj)
#         rec .*= buffer
#         if !isnothing(regularizer)
#             buffer_grad .= ∇reg(rec)
#             rec .-= λ .* buffer_grad
#         end
#         rec .= max.(rec, threshold)  # Ensure non-negativity
#     end

#     # Loss function for logging
#     loss = progress === nothing ? nothing : (myrec -> begin
#         conv_temp!(buffer, myrec, otf)
#         sum(buffer .- measured .* log.(buffer))
#     end)

#     # Logging setup
#     if progress !== nothing
#         record_progress!(progress, rec, 0, loss(rec), 0.0, 1.0)
#     end

#     # Perform iterations
#     for i in 1:iterations
#         iter!(rec)
#         if progress !== nothing
#             record_progress!(progress, copy(rec), i, loss(rec), 0.0, 1.0)
#         end
#     end

#     return rec

# end

# function plan_conv!(u::AbstractArray{T, N}, v::AbstractArray{T, M}, dims=ntuple(+, N)) where {T, N, M}
#     # Retrieve the FFT plan for the given type
#     plan = get_plan(T)

#     # Precompute the forward and inverse FFT plans for `u`
#     P = plan(u, dims)
#     P_inv = inv(P)

#     # Precompute the Fourier transform of `u` and store it
#     u_ft_stor = P * u

#     # Compute the Fourier transform of `v` (PSF or kernel)
#     v_ft = fft_or_rfft(T)(v, dims)

#     # Preallocate the output array for convolution results
#     out = similar(u)

#     # Define the efficient convolution function
#     function conv!(output, input, v_ft=v_ft)
#         p_conv_aux!(P, P_inv, input, v_ft, u_ft_stor, output)
#         return output
#     end

#     return v_ft, conv!
# end

###
# using FFTW, LinearAlgebra, Statistics

function BackProjector(PSF_fp; bp_type="traditional", alpha=0.001, beta=1, n=10, resFlag=1, iRes=[0, 0, 0], verboseFlag=false)
    dims = ndims(PSF_fp)
    Sx, Sy, Sz = dims == 2 ? (size(PSF_fp)..., 1) : size(PSF_fp)
    flippedPSF = reverse(PSF_fp)

    if verboseFlag
        println("Back projector type: $bp_type")
    end

    if bp_type == "traditional"
        PSF_bp = flippedPSF
        OTF_bp = fft(ifftshift(PSF_bp))
        return PSF_bp, OTF_bp
    end

    OTF_flip = fft(ifftshift(flippedPSF))
    OTF_abs = fftshift(abs.(OTF_flip))
    OTFmax = maximum(OTF_abs)

    if bp_type == "wiener"
        OTF_flip_norm = OTF_flip / OTFmax
        OTF_bp = OTF_flip_norm ./ (abs.(OTF_flip_norm).^2 .+ alpha)
        PSF_bp = fftshift(real(ifft(OTF_bp)))
        return PSF_bp, OTF_bp
    end

    resx, resy, resz = resFlag == 0 ? size_to_fwhm(Sx, Sy, Sz) ./ √2 :
                       resFlag == 1 ? size_to_fwhm(Sx, Sy, Sz) :
                       resFlag == 2 ? (dims == 2 ? (iRes[1], iRes[2], 0) : (iRes[1], iRes[2], iRes[3])) :
                       error("Invalid resFlag: $resFlag. Must be 0, 1, or 2.")

    if bp_type == "gaussian"
        PSF_bp = gen_gaussianPSF(Sx, Sy, Sz, resx, resy, resz, dims)
        OTF_bp = fft(ifftshift(PSF_bp))
        return PSF_bp, OTF_bp
    end

    px, py, pz = 1 / Sx, 1 / Sy, 1 / max(1, Sz)
    tx, ty, tz = 1 / (resx * px), 1 / (resy * py), 1 / (resz * pz)

    if verboseFlag
        println("Resolution cutoff (spatial): $resx x $resy x $resz")
        println("Resolution cutoff (Fourier): $tx x $ty x $tz")
    end

    if bp_type == "butterworth"
        return butterworth_filter(Sx, Sy, Sz, tx, ty, tz, beta, n, dims)
    elseif bp_type == "wiener-butterworth"
        OTF_abs_norm = OTF_abs / OTFmax
        return wiener_butterworth_filter(OTF_flip, OTF_abs_norm, alpha, beta, Sx, Sy, Sz, tx, ty, tz, n, dims)
    else
        error("Unsupported bp_type: $bp_type")
    end
end

# Helper function for Gaussian PSF generation (2D or 3D)
function gen_gaussianPSF(Sx, Sy, Sz, FWHMx, FWHMy, FWHMz, dims)
    sigx, sigy, sigz = FWHMx / 2.3548, FWHMy / 2.3548, FWHMz / 2.3548
    if dims == 2
        x = range(-Sx/2, Sx/2, length=Sx)
        y = range(-Sy/2, Sy/2, length=Sy)
        PSF = [exp(-((xi^2 / (2 * sigx^2)) + (yi^2 / (2 * sigy^2)))) for xi in x, yi in y]
    elseif dims == 3
        x = range(-Sx/2, Sx/2, length=Sx)
        y = range(-Sy/2, Sy/2, length=Sy)
        z = range(-Sz/2, Sz/2, length=Sz)
        PSF = [exp(-((xi^2 / (2 * sigx^2)) + (yi^2 / (2 * sigy^2)) + (zi^2 / (2 * sigz^2)))) for xi in x, yi in y, zi in z]
    end
    return PSF ./ sum(PSF) # Normalize
end

# Helper function for Butterworth filter (2D or 3D)
function butterworth_filter(Sx, Sy, Sz, tx, ty, tz, beta, n, dims)
    ee = 1 / beta^2 - 1
    if dims == 2
        kx = range(-Sx/2, Sx/2, length=Sx)
        ky = range(-Sy/2, Sy/2, length=Sy)
        mask = [1 / sqrt(1 + ee * ((kx[i] / tx)^2 + (ky[j] / ty)^2)^n) for i in 1:Sx, j in 1:Sy]
    elseif dims == 3
        kx = range(-Sx/2, Sx/2, length=Sx)
        ky = range(-Sy/2, Sy/2, length=Sy)
        kz = range(-Sz/2, Sz/2, length=Sz)
        mask = [1 / sqrt(1 + ee * ((kx[i] / tx)^2 + (ky[j] / ty)^2 + (kz[k] / tz)^2)^n) for i in 1:Sx, j in 1:Sy, k in 1:Sz]
    end
    OTF_bp = ifftshift(mask)
    PSF_bp = fftshift(real(ifft(OTF_bp)))
    return PSF_bp, OTF_bp
end

# Helper function for Wiener-Butterworth filter (2D or 3D)
function wiener_butterworth_filter(OTF_flip, OTF_abs_norm, alpha, beta, Sx, Sy, Sz, tx, ty, tz, n, dims)
    OTF_flip_norm = OTF_flip ./ maximum(abs.(OTF_flip))
    OTF_Wiener = OTF_flip_norm ./ (abs.(OTF_flip_norm).^2 .+ alpha)
    ee = 1 / beta^2 - 1
    if dims == 2
        kx = range(-Sx/2, Sx/2, length=Sx)
        ky = range(-Sy/2, Sy/2, length=Sy)
        mask = [1 / sqrt(1 + ee * ((kx[i] / tx)^2 + (ky[j] / ty)^2)^n) for i in 1:Sx, j in 1:Sy]
    elseif dims == 3
        kx = range(-Sx/2, Sx/2, length=Sx)
        ky = range(-Sy/2, Sy/2, length=Sy)
        kz = range(-Sz/2, Sz/2, length=Sz)
        mask = [1 / sqrt(1 + ee * ((kx[i] / tx)^2 + (ky[j] / ty)^2 + (kz[k] / tz)^2)^n) for i in 1:Sx, j in 1:Sy, k in 1:Sz]
    end
    mask = ifftshift(mask)
    OTF_bp = mask .* OTF_Wiener
    PSF_bp = fftshift(real(ifft(OTF_bp)))
    return PSF_bp, OTF_bp
end

# Helper to calculate FWHM
function size_to_fwhm(Sx, Sy, Sz)
    return 1.0 / Sx, 1.0 / Sy, 1.0 / max(1, Sz)
end
