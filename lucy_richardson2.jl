
"""
    richardson_lucy_iterative2(measured, psf; <keyword arguments>)

Classical iterative Richardson-Lucy iteration scheme for deconvolution.
`measured` is the measured array and `psf` the point spread function.
Converges slower than the optimization approach of `deconvolution`

# Keyword Arguments
- `psf_bp=nothing`: A psf for backpropagation.
- `regularizer=nothing`: A regularizer function. Can be changed
- `λ=0.05`: A float indicating the total weighting of the regularizer with 
    respect to the global loss function
- `iterations=100`: Specifies number of iterations.
- `progress`: if not `nothing`, the progress will be monitored in a summary dictionary as obtained by
              DeconvOptim.options_trace_deconv()

# Example
```julia-repl
julia> using DeconvOptim, TestImages, Colors, Noise;

julia> img = Float32.(testimage("resolution_test_512"));

julia> psf = Float32.(generate_psf(size(img), 30));

julia> img_b = conv(img, psf);

julia> img_n = poisson(img_b, 300);

julia> @time res = richardson_lucy_iterative(img_n, psf);
```
"""
function richardson_lucy_iterative2(measured, psf; 
                                   psf_bp=nothing,
                                   regularizer=nothing,
                                   λ=0.05,
                                   iterations=100,
                                   conv_dims=1:ndims(psf),
                                   threshold = 0,
                                   progress = nothing)

    otf, conv_temp = plan_conv(measured, psf, conv_dims) 
    # initializer
    ### Different PSF inizialization methods are possible generating custom psf_bp using the relative functions 
    # if psf_bp == nothing
    #     psf_bp = deepcopy(psf)
    #     reverse!(psf_bp)
    # end
    # otf_conj, _ = plan_conv(measured, psf_bp, conv_dims) 
    if psf_bp == nothing
        otf_conj = conj.(otf)
    else
        otf_conj, _ = plan_conv(measured, psf_bp, conv_dims) 
    end
    # Apply threshold
    rec =  map(x -> x >= threshold ? x : threshold, measured) 
    
    #### THIS PART NEED TO BE DEBUGGED!!!
    # buffer for gradient
    # we need Base.invokelatest because of world age issues with generated
    # regularizers
    buffer_grad =  let 
        if !isnothing(regularizer)
            Base.invokelatest(gradient, regularizer, rec)[1]
        else
            nothing
        end
    end

    ∇reg(x) = buffer_grad .= Base.invokelatest(gradient, regularizer, x)[1]

    buffer = copy(measured)

    iter_without_reg(rec) = begin
        buffer .= measured ./ (conv_temp(rec, otf))
        conv_temp(buffer, otf_conj)
    end
    iter_with_reg(rec) = buffer .= (iter_without_reg(rec) .- λ .* ∇reg(rec))

    iter = isnothing(regularizer) ? iter_without_reg : iter_with_reg

    # the loss function is only needed for logging, not for LR itself
    loss(myrec) = begin
        fwd = conv_temp(myrec, otf)
        return sum(fwd .- measured .* log.(fwd))
    end

    # logging part
    tmp_time = 0.0
    if progress !== nothing
        record_progress!(progress, rec, 0, loss(rec), 0.0, 1.0)
        tmp_time=time()
    end
    code_time = 0.0

    # do actual optimization
    for i in 1:iterations
        rec .*= iter(rec)
        if progress !== nothing
            # do not count the time for evaluating the loss here.
            code_time += time() .- tmp_time
            record_progress!(progress, copy(rec), i, loss(rec), code_time, 1.0)
            tmp_time=time()
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
