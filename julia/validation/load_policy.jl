using ForwardNets
using Auto2D

type Tim2DExtractor <: AbstractFeatureExtractor
    feature_means::Vector{Float64} # for standardization
    feature_std::Vector{Float64} # for standardization

    function Tim2DExtractor(filepath::AbstractString,)
        feature_means = vec(h5read(filepath, "initial_obs_mean"))
        feature_std = vec(h5read(filepath, "initial_obs_std"))
        new(feature_means, feature_std)
    end
end

Base.length(::Tim2DExtractor) = N_FEATURES
_standardize(v::Real, μ::Real, σ::Real) = (v - μ)/σ
function AutomotiveDrivingModels.pull_features!{F<:AbstractFloat}(ext::Tim2DExtractor, features::Vector{F}, rec::SceneRecord, roadway::Roadway, vehicle_index::Int)
    pull_features!(features, rec, roadway, vehicle_index)

    # standardize
    for (i,v) in enumerate(features)
        features[i] = _standardize(v, ext.feature_means[i], ext.feature_std[i])
    end
    features
end

function _pull_W_b_h(filepath::AbstractString, path::AbstractString)
    W_xr = h5read(filepath, joinpath(path, "W_xr:0"))::Matrix{Float32}
    W_xu = h5read(filepath, joinpath(path, "W_xu:0"))::Matrix{Float32}
    W_xc = h5read(filepath, joinpath(path, "W_xc:0"))::Matrix{Float32}
    W_x = vcat(W_xr, W_xu, W_xc)
    
    W_hr = h5read(filepath, joinpath(path, "W_hr:0"))::Matrix{Float32}
    W_hu = h5read(filepath, joinpath(path, "W_hu:0"))::Matrix{Float32}
    W_hc = h5read(filepath, joinpath(path, "W_hc:0"))::Matrix{Float32}
    W_h = vcat(W_hr, W_hu, W_hc)
    
    b_r = h5read(filepath, joinpath(path, "b_r:0"))::Vector{Float32}
    b_u = h5read(filepath, joinpath(path, "b_u:0"))::Vector{Float32}
    b_c = h5read(filepath, joinpath(path, "b_c:0"))::Vector{Float32}
    b = vcat(b_r, b_u, b_c)
    
    (W_x, W_h, b)
end

function _pull_W_b(filepath::AbstractString, path::AbstractString)
    W = h5read(filepath, joinpath(path, "W:0"))::Matrix{Float32}
    b = h5read(filepath, joinpath(path, "b:0"))::Vector{Float32}
    (W,b)
end

function load_gru_driver(
            filepath::AbstractString = POLICY_FILEPATH, 
            iteration::Int=-1; 
            gru_layer::Bool=true,
            bc_policy::Bool=false,
            action_type::DataType=AccelTurnrate)

    if iteration == -1 # load most recent
        fid = h5open(filepath)
        for val in keys(read(fid))
            i = parse(Int, match(r"\d+", val).match)
            iteration = max(iteration, i)
        end
    end

    if gru_layer
        basepath = @sprintf("iter%05d/mlp_policy", iteration)
        layers = sort(collect(keys(h5read(filepath, basepath))))
        layers = layers[1:end-1]
    else
        if bc_policy
            basepath = @sprintf("iter%05d/mlp_policy", iteration)
            layers = sort(collect(keys(h5read(filepath, basepath))))
            layers = layers[1:end-2]
        else
            basepath = @sprintf("iter%05d/mlp_policy/mean_network", iteration)
            layers = sort(collect(keys(h5read(filepath, basepath))))
            layers = layers[1:end-1]
        end
    end
    

    W, b = _pull_W_b(filepath, joinpath(basepath, layers[1]))

    net = ForwardNet{Float32}()
    push!(net, Variable(:input, Array(Float32, size(W, 2))))

    # hidden layers
    for layer in layers
        layer_sym = Symbol(layer)
        W, b = _pull_W_b(filepath, joinpath(basepath, layer))
        push!(net, Affine, layer_sym, lastindex(net), size(W, 1))
        copy!(net[layer_sym].W, W)
        copy!(net[layer_sym].b, b)
        push!(net, ELU, Symbol(layer*"ELU"), lastindex(net))
    end

    # Add GRU layer
    if gru_layer
        # output layer
        W, b = _pull_W_b(filepath, joinpath(basepath, "output"))
        push!(net, Affine, :output_layer_mlp, lastindex(net), size(W, 1))
        copy!(net[:output_layer_mlp].W, W)
        copy!(net[:output_layer_mlp].b, b)
        push!(net, Variable(:output_mlp, output(net[:output_layer_mlp])), lastindex(net))
        push!(net, ELU, Symbol("output_ELU"), lastindex(net))

        basepath = @sprintf("iter%05d/gru_policy", iteration)
        W_x, W_h, b = _pull_W_b_h(filepath, joinpath(basepath, "mean_network", "gru"))
        push!(net, GRU, :gru, :output_mlp, size(W_h, 2))
        copy!(net[:gru].W_x, W_x)
        copy!(net[:gru].W_h, W_h)
        copy!(net[:gru].b, b)
        copy!(net[:gru].h_prev, zeros(size(W_h, 2)))
        push!(net, Variable(:output_gru, output(net[:gru])), lastindex(net))

        # Finally, output layer from GRU
        W, b = _pull_W_b(filepath, joinpath(basepath, "mean_network", "output_flat"))
        push!(net, Affine, :output_layer, :gru, size(W, 1))
        copy!(net[:output_layer].W, W)
        copy!(net[:output_layer].b, b)
        push!(net, Variable(:output, output(net[:output_layer])), lastindex(net))
    else
        # Output layer
        if bc_policy
            W, b = _pull_W_b(filepath, joinpath(basepath, "mean_network", "output_flat"))
        else
            W, b = _pull_W_b(filepath, joinpath(basepath, "output"))
        end
        push!(net, Affine, :output_layer, lastindex(net), size(W, 1))
        copy!(net[:output_layer].W, W)
        copy!(net[:output_layer].b, b)
        push!(net, Variable(:output, output(net[:output_layer])), lastindex(net))
        basepath = @sprintf("iter%05d/mlp_policy", iteration)
    end

    if gru_layer
        logstdevs = vec(h5read(filepath, joinpath(basepath, "output_log_std/param:0")))::Vector{Float32}
    else
        if bc_policy
            logstdevs = vec(h5read(filepath, joinpath(basepath, "output_log_std/param:0")))::Vector{Float32}
        else
            logstdevs = vec(h5read(filepath, joinpath(basepath, "output_std_param/param:0")))::Vector{Float32}
        end
    end
    Σ = convert(Vector{Float64}, exp(logstdevs))
    Σ = Σ.^2

    # extactor
    extractor = Tim2DExtractor(joinpath(ROOT_FILEPATH, "julia", "validation", "models", "gail_gru.h5"))

    Auto2D.GaussianMLPDriver(action_type, net, extractor, IntegratedContinuous(0.1,1),
                        input = :input, output = :output, Σ = Σ)
end
