using DeconvOptim: gradient

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
function deconvRL(measured, psf; 
                                   psf_bp=nothing,
                                   regularizer=nothing,
                                   λ=0.05,
                                   iterations=100,
                                   conv_dims=1:ndims(psf),
                                   threshold=0,
                                   progress=nothing)

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

    buffer = similar(measured)

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


###
# using FFTW, LinearAlgebra, Statistics

function BackProjector(PSF_fp; bp_type="traditional", alpha=0.001, beta=1, n=10, resFlag=1, iRes=[0, 0, 0], verboseFlag=false)
    # Initialize dimensions
    dims = ndims(PSF_fp)
    if dims == 2
        Sx, Sy = size(PSF_fp)
        Sz = 1
    elseif dims == 3
        Sx, Sy, Sz = size(PSF_fp)
    else
        error("Input PSF must be 2D or 3D")
    end

    # Scx, Scy, Scz = (Sx + 1) / 2, (Sy + 1) / 2, (Sz + 1) / 2

    if verboseFlag
        println("Back projector type: $bp_type")
    end

    # Flip PSF
    flippedPSF = deepcopy(PSF_fp)
    reverse!(flippedPSF)

    if bp_type == "traditional"
        PSF_bp = flippedPSF
        OTF_bp = fft(ifftshift(PSF_bp))

    else

        # Fourier Transform of flipped PSF
        OTF_flip = fft(ifftshift(flippedPSF))
        OTF_abs = fftshift(abs.(OTF_flip))
        OTFmax = maximum(OTF_abs)

        if bp_type == "wiener"
            OTF_flip_norm = OTF_flip / OTFmax
            OTF_bp = OTF_flip_norm ./ (abs.(OTF_flip_norm).^2 .+ alpha)
            PSF_bp = fftshift(real(ifft(OTF_bp)))
        else

            # Resolution cutoff
            resx, resy, resz = if resFlag == 0
                FWHMx, FWHMy, FWHMz = size_to_fwhm(Sx, Sy, Sz)
                (FWHMx / √2, FWHMy / √2, FWHMz / √2)
            elseif resFlag == 1
                size_to_fwhm(Sx, Sy, Sz)
            elseif resFlag == 2
                dims == 2 ? (iRes[1], iRes[2], 0) : (iRes[1], iRes[2], iRes[3])
            else
                error("Invalid resFlag: $resFlag. Must be 0, 1, or 2.")
            end

            if bp_type == "gaussian"
                PSF_bp = gen_gaussianPSF(Sx, Sy, Sz, resx, resy, resz, dims)
                OTF_bp = fft(ifftshift(PSF_bp))
            else

                # Pixel size and frequency cutoff
                px, py, pz = 1 / Sx, 1 / Sy, 1 / max(1, Sz)
                tx, ty, tz = 1 / (resx * px), 1 / (resy * py), 1 / (resz * pz)

                if verboseFlag
                    println("Resolution cutoff (spatial): $resx x $resy x $resz")
                    println("Resolution cutoff (Fourier): $tx x $ty x $tz")
                end

                PSF_bp, OTF_bp = nothing, nothing

                # Process each back projector type
                
                if bp_type == "butterworth"
                    PSF_bp, OTF_bp = butterworth_filter(Sx, Sy, Sz, tx, ty, tz, beta, n, dims)
                elseif bp_type == "wiener-butterworth"
                    OTF_abs_norm = OTF_abs / OTFmax
                    PSF_bp, OTF_bp = wiener_butterworth_filter(OTF_flip, OTF_abs_norm, alpha, beta, Sx, Sy, Sz, tx, ty, tz, n, dims)
                else
                    error("Unsupported bp_type: $bp_type")
                end
            end
        end
    end

    return convert(typeof(PSF_fp), PSF_bp), OTF_bp
end

# Helper function for Gaussian PSF generation (2D or 3D)
function gen_gaussianPSF(Sx, Sy, Sz, FWHMx, FWHMy, FWHMz, dims)
    sigx, sigy, sigz = FWHMx / 2.3548, FWHMy / 2.3548, FWHMz / 2.3548
    if dims == 2
        x = range(-Sx/2, Sx/2, length=Sx)
        y = range(-Sy/2, Sy/2, length=Sy)
        X, Y = ndgrid(x, y)
        PSF = exp.(-((X.^2 / (2 * sigx^2)) .+ (Y.^2 / (2 * sigy^2))))
    elseif dims == 3
        x = range(-Sx/2, Sx/2, length=Sx)
        y = range(-Sy/2, Sy/2, length=Sy)
        z = range(-Sz/2, Sz/2, length=Sz)
        X, Y, Z = ndgrid(x, y, z)
        PSF = exp.(-((X.^2 / (2 * sigx^2)) .+ (Y.^2 / (2 * sigy^2)) .+ (Z.^2 / (2 * sigz^2))))
    end
    return PSF / sum(PSF) # Normalize
end

# Helper function for Butterworth filter (2D or 3D)
function butterworth_filter(Sx, Sy, Sz, tx, ty, tz, beta, n, dims)
    ee = 1 / beta^2 - 1
    if dims == 2
        kx = range(-Sx/2, Sx/2, length=Sx)
        ky = range(-Sy/2, Sy/2, length=Sy)
        KX, KY = ndgrid(kx, ky)
        mask = 1 ./ sqrt.(1 .+ ee .* ((KX / tx).^2 .+ (KY / ty).^2).^n)
    elseif dims == 3
        kx = range(-Sx/2, Sx/2, length=Sx)
        ky = range(-Sy/2, Sy/2, length=Sy)
        kz = range(-Sz/2, Sz/2, length=Sz)
        KX, KY, KZ = ndgrid(kx, ky, kz)
        mask = 1 ./ sqrt.(1 .+ ee .* ((KX / tx).^2 .+ (KY / ty).^2 .+ (KZ / tz).^2).^n)
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
        KX, KY = ndgrid(kx, ky)
        mask = 1 ./ sqrt.(1 .+ ee .* ((KX / tx).^2 .+ (KY / ty).^2).^n)
    elseif dims == 3
        kx = range(-Sx/2, Sx/2, length=Sx)
        ky = range(-Sy/2, Sy/2, length=Sy)
        kz = range(-Sz/2, Sz/2, length=Sz)
        KX, KY, KZ = ndgrid(kx, ky, kz)
        mask = 1 ./ sqrt.(1 .+ ee .* ((KX / tx).^2 .+ (KY / ty).^2 .+ (KZ / tz).^2).^n)
    end
    mask = ifftshift(mask)
    OTF_bp = mask .* OTF_Wiener
    PSF_bp = fftshift(real(ifft(OTF_bp)))
    return PSF_bp, OTF_bp
end

# Helper to calculate FWHM
function size_to_fwhm(Sx, Sy, Sz)
    FWHMx = 1.0 / Sx
    FWHMy = 1.0 / Sy
    FWHMz = 1.0 / max(1, Sz)
    return FWHMx, FWHMy, FWHMz
end

# Helper to generate grids of coordinates for multi-dimensional spaces.
function ndgrid(v1, v2, v3=nothing)
    if isnothing(v3)
        # 2D case
        X = reshape(v1, :, 1) .* ones(Float32, 1, length(v2))  # Create the grid for v1
        Y = ones(Float32, length(v1), 1) .* reshape(v2, 1, :)  # Create the grid for v2
        return X, Y
    else
        # 3D case
        X = reshape(v1, :, 1, 1) .* ones(Float32, 1, length(v2), length(v3))  # Create grid for v1
        Y = ones(Float32, length(v1), 1, 1) .* reshape(v2, 1, :, 1)          # Create grid for v2
        Z = ones(Float32, length(v1), length(v2), 1) .* reshape(v3, 1, 1, :) # Create grid for v3
        return X, Y, Z
    end
end
