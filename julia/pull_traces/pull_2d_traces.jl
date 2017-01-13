using AutomotiveDrivingModels
using NGSIM
using HDF5, JLD

ΔT = NGSIM_TIMESTEP # [s]

immutable Interval
    id::Int
    frame_lo::Int
    frame_hi::Int
end
type IntervalInProgress
    id::Int
    frame_lo::Int
    updated::Bool
end

# include("tim2dfeatures.jl")
# include("radar_features.jl")
include("multifeatureset.jl")

extract_core = true
extract_temporal = true
extract_well_behaved = false
extract_neighbor_features = false
extract_carlidar_rangerate = true
carlidar_nbeams = 20
roadlidar_nbeams = 0
roadlidar_nlanes = 0
carlidar_max_range = 100.0 # [m]
roadlidar_max_range = 50.0 # [m]

extractor = MultiFeatureExtractor(
    extract_core,
    extract_temporal,
    extract_well_behaved,
    extract_neighbor_features,
    extract_carlidar_rangerate,
    carlidar_nbeams,
    roadlidar_nbeams,
    roadlidar_nlanes,
    carlidar_max_range=carlidar_max_range,
    roadlidar_max_range=roadlidar_max_range,
    )

N_FEATURES = length(extractor)

# const ROADWAY = open(io->read(io, Roadway), "roadway_passive_aggressive.txt", "r")
# for trajdata_path in ["trajdata_passive_aggressive1.txt"]
# println("FILE: ", trajdata_path); tic()
#     trajdata = open(io->read(io, Trajdata, ROADWAY), trajdata_path, "r")

culling_fidelity = 5.0 # [m]
lane_portion_max_range = roadlidar_max_range + culling_fidelity/2 + 5.0 # [m] note extra margin of error

err = 0.0

for trajdata_path in TRAJDATA_PATHS
    println("FILE: ", trajdata_path); tic()
    trajdata = load_trajdata(trajdata_path)

    extractor.road_lidar_culling = RoadwayLidarCulling(trajdata.roadway, lane_portion_max_range, culling_fidelity)
    @assert extractor.road_lidar_culling.is_leaf == false

    segments = pull_continuous_segments(trajdata, 1)

    # extract the dataset for each segment
    scene = Scene()
    rec = SceneRecord(2, ΔT)
    n_states = sum(i->length(i.frame_lo:i.frame_hi)-1, segments)
    features = Array(Float64, N_FEATURES)
    featureset = Array(Float64, length(features), n_states)
    targetset = Array(Float64, 2, n_states)
    interval_starts = Array(Int, length(segments))

    i = 0
    for (segind, seg) in enumerate(segments)
        empty!(rec)
        interval_starts[segind] = i+1
        tic()
        for frame in seg.frame_lo : seg.frame_hi
            get!(scene, trajdata, frame)
            update!(rec, scene)
            vehicle_index = get_index_of_first_vehicle_with_id(rec, seg.egoid)
            @assert(vehicle_index != 0)
            pull_features!(extractor, features, rec, trajdata.roadway, vehicle_index)

            if frame > seg.frame_lo
                targetset[1, i] = get(ACC, rec, trajdata.roadway, vehicle_index)
                targetset[2, i] = get(TURNRATEG, rec, trajdata.roadway, vehicle_index)
            end

            if frame < seg.frame_hi
                i += 1
                featureset[:,i] = features
            end
        end
        toc()
    end

    # Save it all to a file
    println("saving...")
    outfile = ("./../2d_drive_data/data_" * splitdir(splitext(trajdata_path)[1])[2]
    * "_clb" * string(carlidar_nbeams)
    * "_rlb" * string(roadlidar_nbeams)
    * "_rll" * string(roadlidar_nlanes)
    * "_clmr" * string(Int(carlidar_max_range))
    * "_rlmr" * string(Int(roadlidar_max_range))
    * ".jld")

    #outfile = "./data_" * splitdir(splitext(trajdata_path)[1])[2] * ".jld"
    JLD.save(outfile, "features", featureset, "targets", targetset, "intervals", interval_starts, "timestep", ΔT)
    toc()
end
