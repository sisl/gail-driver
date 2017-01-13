using AutomotiveDrivingModels
using NGSIM
using HDF5

include("RootDir.jl")
const TRAIN_TEST_SPLIT_FILEPATH = joinpath(ROOT_FILEPATH, "data", "NGSIM_train_test_split.h5")

export
    load_assignment,
    load_evaldata

function load_assignment(filepath::AbstractString = TRAIN_TEST_SPLIT_FILEPATH)
    h5read(filepath, "data/assignment")
end

function load_evaldata(filepath::AbstractString = TRAIN_TEST_SPLIT_FILEPATH)
    trajdata_indeces = h5read(filepath, "data/trajdata_indeces")
    egoids = h5read(filepath, "data/egoids")
    frame_lo = h5read(filepath, "data/frame_lo")
    frame_hi = h5read(filepath, "data/frame_hi")

    trajdatas = map(i->load_trajdata(i), 1:length(TRAJDATA_PATHS))
    segments = map(i->TrajdataSegment(trajdata_indeces[i], egoids[i], frame_lo[i], frame_hi[i]), 1:length(trajdata_indeces))

    EvaluationData(trajdatas, segments)
end
