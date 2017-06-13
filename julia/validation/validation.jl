include("load_train_test_split.jl")

using AutoDrivers
using AutoDrivers.GaussianMixtureRegressionDrivers

include("../envs/Auto2D.jl")
include("load_policy.jl")
include("additional_metrics.jl")

const METRIC_SAVE_FILE_DIR = dirname(@__FILE__) * "/results/"

const N_SEGMENTS = 1000
const N_SIMULATIONS_PER_TRACE = 20
const EVAL_PRIME_DURATION = 0.0 # [s] - length of priming period
const EVAL_PRIME_STEPS = 0
const EVAL_DURATION = 10.0 # [s] - length of trace
const EVAL_DURATION_STEPS = 100
const CONTEXT = IntegratedContinuous(NGSIM_TIMESTEP, 3)
const METRICS = TraceMetricExtractor[
    RootWeightedSquareError(SPEED,  0.5),
    RootWeightedSquareError(SPEED,  1.0),
    RootWeightedSquareError(SPEED,  2.0),
    RootWeightedSquareError(SPEED,  3.0),
    RootWeightedSquareError(SPEED,  4.0),
    RootWeightedSquareError(SPEED,  5.0),

    RootWeightedSquareError(POSFT,  0.5),
    RootWeightedSquareError(POSFT,  1.0),
    RootWeightedSquareError(POSFT,  2.0),
    RootWeightedSquareError(POSFT,  3.0),
    RootWeightedSquareError(POSFT,  4.0),
    RootWeightedSquareError(POSFT,  5.0),

    RWSEPosG(0.5),
    RWSEPosG(1.0),
    RWSEPosG(2.0),
    RWSEPosG(3.0),
    RWSEPosG(4.0),
    RWSEPosG(5.0),

    EmergentKLDivergence(INV_TTC, 0., 10., 100),
    EmergentKLDivergence(SPEED, -5., 50., 100),
    EmergentKLDivergence(ACC, -10., 10., 100),
    EmergentKLDivergence(TURNRATEF, -2., 2., 100),
    EmergentKLDivergence(JERK, -100., 100., 100),

    SumSquareJerk(),
    CollisionRate(),
    OffRoadRate(),
    LaneChangeRate(),
    HardBrakeRate(),
]

const EXTRACT_CORE = true
const EXTRACT_TEMPORAL = false
const EXTRACT_WELL_BEHAVED = true
const EXTRACT_NEIGHBOR_FEATURES = false
const EXTRACT_CARLIDAR_RANGERATE = true
const CARLIDAR_NBEAMS = 20
const ROADLIDAR_NBEAMS = 0
const ROADLIDAR_NLANES = 2
const CARLIDAR_MAX_RANGE = 100.0
const ROADLIDAR_MAX_RANGE = 100.0

evaldata = load_evaldata()
assignment = load_assignment()

function create_evaldata(evaldata::EvaluationData, foldset::FoldSet; nsegs::Int=1, nframes::Int=101)
    segments = Array(TrajdataSegment, nsegs)
    indeces = collect(foldset)
    i = 0
    # seg_exclude = [68] # Array of trajectories to exclude
    while i < nsegs
        seg = evaldata.segments[indeces[rand(1:length(indeces))]]

        # Make sure segment is long enough and not in first or last 500 frames
        frame_lo = max(seg.frame_lo, 500)
        frame_hi = min(seg.frame_hi, AutoCore.nframes(evaldata.trajdatas[seg.trajdata_index]) - 1000)
        if frame_hi - frame_lo + 1 > nframes #&& !(seg.trajdata_index in seg_exclude))
            dom_hi = (frame_hi - frame_lo + 1) - nframes
            domain = 1:dom_hi

            frame_lo += rand(domain)
            frame_hi = frame_lo + nframes - 1

            segments[i+=1] = TrajdataSegment(seg.trajdata_index, seg.egoid, frame_lo, frame_hi)
        end
    end

    EvaluationData(evaldata.trajdatas, segments)
end

function create_simparams(evaldata::EvaluationData)
    # Construct extractor
    extractor = Auto2D.MultiFeatureExtractor(
        EXTRACT_CORE,
        EXTRACT_TEMPORAL,
        EXTRACT_WELL_BEHAVED,
        EXTRACT_NEIGHBOR_FEATURES,
        EXTRACT_CARLIDAR_RANGERATE,
        CARLIDAR_NBEAMS,
        ROADLIDAR_NBEAMS,
        ROADLIDAR_NLANES,
        carlidar_max_range=CARLIDAR_MAX_RANGE,
        roadlidar_max_range=ROADLIDAR_MAX_RANGE,
        )

    # Convert array of trajdatas to dict
    trajdatas = Dict(zip(collect(1:length(evaldata.trajdatas)), evaldata.trajdatas))

    # Construct and return simparams
    Auto2D.SimParams(trajdatas, evaldata.segments, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        false, true, false, -2.0, 1, EVAL_PRIME_STEPS, EVAL_DURATION_STEPS, AccelTurnrate, extractor)
end

srand(0)
eval_seg_nframes = ceil(Int, (EVAL_PRIME_DURATION + EVAL_DURATION)/NGSIM_TIMESTEP) + 1
VALDATA_SUBSET = create_evaldata(evaldata, foldset_match(assignment, FOLD_TEST), nsegs=N_SEGMENTS, nframes=eval_seg_nframes)
FOLDSET_TEST = foldset_match(fill(1, N_SEGMENTS), 1)
SIMPARAMS = create_simparams(VALDATA_SUBSET)

function load_models(; context::IntegratedContinuous = CONTEXT)
    models = Dict{AbstractString, DriverModel}()

    models["SG"] = StaticGaussianDriver{AccelTurnrate}(context, MvNormal([0.07813232200000027,0.0025751835870002756], [[0.533053, 0.000284046] [0.000284046, 0.000348645]]))

    mlon = IntelligentDriverModel(
        σ=0.1,
        k_spd=1.0,
        T=0.5,
        s_min=2.0,
        a_max=3.0,
        d_cmf=2.5,
        )
    mlat = ProportionalLaneTracker(σ=0.1, kp=3.0, kd=2.0)
    mlane = MOBIL(context,
                  politeness=0.1,
                  advantage_threshold=0.01,
                 )

    models["controller"] = Tim2DDriver(context, mlon=mlon, mlat=mlat, mlane=mlane, rec=SceneRecord(3, context.Δt))

    include(joinpath(ROOT_FILEPATH, "julia/pull_traces", "multifeatureset.jl"))
    extractor = MultiFeatureExtractor(EXTRACT_CORE, EXTRACT_TEMPORAL, 
                                    EXTRACT_WELL_BEHAVED, EXTRACT_NEIGHBOR_FEATURES, 
                                    EXTRACT_CARLIDAR_RANGERATE, CARLIDAR_NBEAMS,
                                    ROADLIDAR_NBEAMS, ROADLIDAR_NLANES)
    models["GMR"] = open(io->read(io, GaussianMixtureRegressionDriver, extractor), "GMR.txt", "r")

    filepath = joinpath(ROOT_FILEPATH, "julia", "validation",  "models", "gail_gru.h5")
    iteration = 413
    models["gail_gru"] = Auto2D.load_gru_driver(filepath, iteration)

    filepath = "./models/gail_mlp.h5"
    iteration = 447
    models["gail_mlp"] = load_gru_driver(filepath, iteration, gru_layer=false)

    filepath = "./models/bc_gru.h5"
    iteration = -1
    models["bc_gru"] = Auto2D.load_gru_driver(filepath, iteration, bc_policy=true)

    filepath = "./models/bc_mlp.h5"
    iteration = -1
    models["bc_mlp"] = load_gru_driver(filepath, iteration, gru_layer=false, bc_policy=true)

    models
end

function validate(model::DriverModel;
    simparams::Auto2D.SimParams = SIMPARAMS,
    metrics::Vector{TraceMetricExtractor} = METRICS,
    foldset::FoldSet = FOLDSET_TEST,
    n_simulations_per_trace::Int = N_SIMULATIONS_PER_TRACE,
    save::Bool=false,
    modelname::AbstractString=AutomotiveDrivingModels.get_name(model)
    )

    metrics_df = allocate_metrics_dataframe(METRICS, 1)
    calc_metrics!(metrics_df, model, metrics, simparams, foldset,
                        n_simulations_per_trace=n_simulations_per_trace,
                        row = 1, prime_history=EVAL_PRIME_STEPS)

    if save
        filename = "valid_"*modelname*".csv"
        writetable(joinpath(METRIC_SAVE_FILE_DIR, filename), metrics_df)
    end

    metrics_df
end

print("DONE!")

