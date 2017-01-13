module Auto2D

using AutomotiveDrivingModels
using AutoViz
using NGSIM
import Reel

export gen_simparams, reset, tick, reward, observe, step, isdone, action_space_bounds, observation_space_bounds, render
export SimParams, reel_drive

##################################

include("../validation/load_train_test_split.jl")
include("../pull_traces/multifeatureset.jl")
function get_train_segments(trajdatas::Dict{Int, Trajdata}, nsteps::Int)

    assignment = load_assignment()
    evaldata = load_evaldata()

    # pull all segments that are for training
    all_train_segments = evaldata.segments[find(assignment .== FOLD_TRAIN)]

    # only keep segments that are long enough and are not within 500 frames of start or end
    train_segments = TrajdataSegment[]
    for seg in all_train_segments
        if !haskey(trajdatas, seg.trajdata_index) # only keep valid segments
            continue
        end
        trajdata = trajdatas[seg.trajdata_index]
        frame_lo = max(seg.frame_lo, 500)
        frame_hi = min(seg.frame_hi, nframes(trajdata) - 500)
        if frame_hi - frame_lo > nsteps # if it is long enough
            push!(train_segments, TrajdataSegment(seg.trajdata_index, seg.egoid, frame_lo, frame_hi))
        end
    end

    train_segments
end

##################################
# SimParams

type SimState
    frame::Int
    start_frame::Int
    egoid::Int
    trajdata_index::Int

    scene::Scene
    rec::SceneRecord

    playback_reactive_active_vehicle_ids::Set{Int}
    playback_reactive_speeds::Dict{Int, Float64}

    function SimState(context::IntegratedContinuous, rec_size::Int)
        retval = new()
        retval.scene = Scene()
        retval.rec = SceneRecord(rec_size, context.Δt)
        retval.playback_reactive_active_vehicle_ids = Set{Int}()
        retval.playback_reactive_speeds = Dict{Int, Float64}()
        retval
    end
end

type SimParams
    col_weight::Float64 # reward for collision
    off_weight::Float64 # reward for being offroad
    rev_weight::Float64 # reward for reversal (negative speeds)
    jrk_weight::Float64 # reward scale for square jerk
    acc_weight::Float64 # reward scale for square acceleration
    cen_weight::Float64 # reward scale for lane center offset
    ome_weight::Float64 # reward scale for turn acceleration (omega)
    use_debug_reward::Bool # false by default

    context::IntegratedContinuous
    prime_history::Int
    ego_action_type::DataType

    use_playback_reactive::Bool
    playback_reactive_model::DriverModel{LatLonAccel, IntegratedContinuous}
    playback_reactive_threshold_brake::Float64
    playback_reactive_scene_buffer::Scene

    trajdatas::Dict{Int, Trajdata}
    segments::Vector{TrajdataSegment}

    nsteps::Int
    step_counter::Int
    simstates::Vector{SimState}
    features::Vector{Float64}
    extractor::MultiFeatureExtractor
end
function SimParams(trajdatas::Dict{Int, Trajdata}, segments::Vector{TrajdataSegment},
    col_weight::Float64,
    off_weight::Float64,
    rev_weight::Float64,
    jrk_weight::Float64,
    acc_weight::Float64,
    cen_weight::Float64,
    ome_weight::Float64,
    use_debug_reward::Bool,
    use_playback_reactive::Bool,
    playback_reactive_threshold_brake::Float64, # [m/s²]
    nsimstates::Int,
    prime_history::Int,
    nsteps::Int,
    ego_action_type::DataType,
    extractor::MultiFeatureExtractor,
    context = IntegratedContinuous(NGSIM_TIMESTEP,1),
    )

    simstates = Array(SimState, nsimstates)
    for i in 1 : length(simstates)
        simstates[i] = SimState(context, prime_history+1)
    end

    features = Array(Float64, length(extractor))

    playback_reactive_model = LatLonSeparableDriver(
        context,
        ProportionalLaneTracker(σ=0.1, kp=3.0, kd=2.0),
        IntelligentDriverModel(σ=0.1, k_spd=1.0, T=0.5, s_min=1.0, a_max=3.0, d_cmf=2.5),
        )
    playback_reactive_scene_buffer = Scene()

    SimParams(
        col_weight, off_weight, rev_weight, jrk_weight, acc_weight, cen_weight, ome_weight, use_debug_reward,
        context, prime_history, ego_action_type,
        use_playback_reactive, playback_reactive_model,
        playback_reactive_threshold_brake, playback_reactive_scene_buffer,
        trajdatas, segments, nsteps, 0, simstates, features, extractor)
end
function gen_simparams(trajdata_indeces::Vector,
    col_weight::Float64,
    off_weight::Float64,
    rev_weight::Float64,
    jrk_weight::Float64,
    acc_weight::Float64,
    cen_weight::Float64,
    ome_weight::Float64,
    use_debug_reward::Bool,
    use_playback_reactive::Bool,
    playback_reactive_threshold_brake::Float64,
    nsimstates::Int,
    prime_history::Int,
    nsteps::Int,
    ego_action_type::DataType,
    extractor::MultiFeatureExtractor,
    )

    println("loading trajdatas: ", trajdata_indeces); tic()
    trajdatas = Dict{Int, Trajdata}()
    for i in trajdata_indeces
        trajdatas[i] = load_trajdata(i)
    end
    toc()

    println("loading training segments"); tic()
    segments = get_train_segments(trajdatas, nsteps)
    toc()

    SimParams(trajdatas, segments, col_weight, off_weight, rev_weight, jrk_weight, acc_weight, cen_weight, ome_weight,
              use_debug_reward, use_playback_reactive, playback_reactive_threshold_brake,
              nsimstates, prime_history, nsteps, ego_action_type, extractor)
end
function gen_simparams_from_trajdatas(trajdata_filepaths::Vector, roadway_filepaths::Vector,
    col_weight::Float64,
    off_weight::Float64,
    rev_weight::Float64,
    jrk_weight::Float64,
    acc_weight::Float64,
    cen_weight::Float64,
    ome_weight::Float64,
    use_debug_reward::Bool,
    use_playback_reactive::Bool,
    playback_reactive_threshold_brake::Float64,
    nsimstates::Int,
    prime_history::Int,
    nsteps::Int,
    ego_action_type::DataType,
    extractor::MultiFeatureExtractor,
    )

    println("loading trajdatas"); tic()
    roadways = Dict{String, Roadway}()
    trajdatas = Dict{Int, Trajdata}()
    for (i,filepath) in enumerate(trajdata_filepaths)
        roadwayfp = roadway_filepaths[i]
        if !haskey(roadways, roadwayfp)
            roadways[roadwayfp] = open(io->read(io, Roadway), roadwayfp, "r")
        end
        trajdatas[i] = open(io->read(io, Trajdata, roadways[roadwayfp]), filepath, "r")
    end
    toc()

    println("generating training segments"); tic()
    segments = TrajdataSegment[]
    for (trajdata_index,trajdata) in trajdatas
        append!(segments, pull_continuous_segments(trajdata, trajdata_index))
    end
    toc()

    SimParams(trajdatas, segments, col_weight, off_weight, rev_weight, jrk_weight, acc_weight, cen_weight, ome_weight,
              use_debug_reward, use_playback_reactive, playback_reactive_threshold_brake,
              nsimstates, prime_history, nsteps, ego_action_type, extractor)
end
function gen_simparams(batch_size::Int, args::Dict)

    col_weight = get(args, "col_weight", -2.0)
    off_weight = get(args, "off_weight", -0.75)
    rev_weight = get(args, "rev_weight", -0.50)
    jrk_weight = get(args, "jrk_weight", -0.050)
    acc_weight = get(args, "acc_weight", -0.050)
    cen_weight = get(args, "cen_weight", -0.050)
    ome_weight = get(args, "ome_weight", -0.500)
    use_debug_reward = get(args, "use_debug_reward", false)
    use_playback_reactive = convert(Bool, get(args, "use_playback_reactive", false))
    playback_reactive_threshold_brake = get(args, "playback_reactive_threshold_brake", -2.0) # [m/s²]
    prime_history = get(args, "prime_history", 2)
    nsteps = get(args, "nsteps", 100)

    extract_core = get(args, "extract_core", true)
    extract_temporal = get(args, "extract_temporal", true)
    extract_well_behaved = get(args, "extract_well_behaved", true)
    extract_neighbor_features = get(args, "extract_neighbor_features", false)
    extract_carlidar_rangerate = get(args, "extract_carlidar_rangerate", true)
    carlidar_nbeams = get(args, "carlidar_nbeams", 0)
    roadlidar_nbeams = get(args, "roadlidar_nbeams", 0)
    roadlidar_nlanes = get(args, "roadlidar_nlanes", 2)
    carlidar_max_range = get(args, "carlidar_max_range", 100.0) # [m]
    roadlidar_max_range = get(args, "roadlidar_max_range", 100.0) # [m]

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

    ego_action_type = AccelTurnrate
    T = get(args, "action_type", "AccelTurnrate")
    if T == "AccelTurnrate"
        ego_action_type = AccelTurnrate
    elseif T == "LatLonAccel"
        ego_action_type = LatLonAccel
    end

    if haskey(args, "trajdata_filepaths")
        trajdata_filepaths = args["trajdata_filepaths"]
        roadway_filepaths = args["roadway_filepaths"]
        gen_simparams_from_trajdatas(trajdata_filepaths, roadway_filepaths,
            col_weight, off_weight, rev_weight, jrk_weight, acc_weight, cen_weight, ome_weight,
            use_debug_reward, use_playback_reactive, playback_reactive_threshold_brake,
            batch_size, prime_history, nsteps, ego_action_type, extractor)
    else
        trajdata_indeces = get(args, "trajdata_indeces", [1,2,3,4,5,6])
        gen_simparams(trajdata_indeces,
            col_weight, off_weight, rev_weight, jrk_weight, acc_weight, cen_weight, ome_weight,
            use_debug_reward, use_playback_reactive, playback_reactive_threshold_brake,
            batch_size, prime_history, nsteps, ego_action_type, extractor)
    end
end

Base.show(io::IO, simparams::SimParams) = print(io, "SimParams")

function restart!(simstate::SimState, simparams::SimParams)

    # pick a random segment
    train_seg_index = rand(1:length(simparams.segments))
    seg = simparams.segments[train_seg_index]
    simstate.egoid = seg.egoid
    simstate.trajdata_index = seg.trajdata_index

    # pull the trajdata
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    # pick a random sub-trajectory
    candidate_frame_lo = seg.frame_lo
    candidate_frame_hi = seg.frame_hi - simparams.nsteps - simparams.prime_history
    simstate.frame = rand(candidate_frame_lo:candidate_frame_hi)

    # clear rec and make first observations
    empty!(simstate.rec)
    for i in 1 : simparams.prime_history + 1
        get!(simstate.scene, trajdata, simstate.frame)
        update!(simstate.rec, simstate.scene)
        simstate.frame += 1
    end
    simstate.frame -= 1
    simstate.start_frame = simstate.frame

    # empty playback reactive
    empty!(simstate.playback_reactive_active_vehicle_ids)

    simstate
end

function Base.reset(simparams::SimParams)
    for state in simparams.simstates
        restart!(state, simparams)
    end
    simparams.step_counter = 0
    simparams
end

function step_forward!(simstate::SimState, simparams::SimParams, action_ego::Vector{Float64})

    # propagate the ego vehicle
    trajdata = simparams.trajdatas[simstate.trajdata_index]
    veh_ego = get_vehicle(simstate.scene, simstate.egoid)
    action_ego = convert(simparams.ego_action_type, action_ego)
    ego_state = propagate(veh_ego, action_ego, simparams.context, trajdata.roadway)

    simstate.frame += 1
    if simparams.use_playback_reactive

        model = simparams.playback_reactive_model

        empty!(simparams.playback_reactive_scene_buffer)

        tdframe = trajdata.frames[simstate.frame]
        for i in tdframe.lo : tdframe.hi
            s = trajdata.states[i]
            if s.id != simstate.egoid

                use_playback = true
                veh_index = get_index_of_first_vehicle_with_id(simstate.scene, s.id)
                if veh_index != 0 # iscarinframe

                    if !in(s.id, simstate.playback_reactive_active_vehicle_ids)
                        simstate.playback_reactive_speeds[s.id] = simstate.scene[veh_index].state.v
                    end

                    set_desired_speed!(model, simstate.playback_reactive_speeds[s.id])
                    AutomotiveDrivingModels.observe!(model, simstate.scene, trajdata.roadway, s.id)
                    action = rand(model)::LatLonAccel

                    if s.id in simstate.playback_reactive_active_vehicle_ids ||
                       action.a_lon < simparams.playback_reactive_threshold_brake

                        # use IDM
                        use_playback = false
                        push!(simstate.playback_reactive_active_vehicle_ids, s.id)

                        veh = simstate.scene[veh_index]
                        veh_state = propagate(veh, action, simparams.context, trajdata.roadway)
                        veh2 = Vehicle(veh_state, veh.def)
                        push!(simparams.playback_reactive_scene_buffer, veh2)
                    end
                end

                if use_playback
                    # if it is a new car not in previous frame or it doesn't need IDM
                    veh = Vehicle(s.state, get_vehicledef(trajdata, s.id))
                    push!(simparams.playback_reactive_scene_buffer, veh)
                end
            end
        end

        # move simparams.playback_reactive_scene_buffer over to simparams.scene
        # and place new ego pos
        ego_def = veh_ego.def
        copy!(simstate.scene, simparams.playback_reactive_scene_buffer)
        push!(simstate.scene, Vehicle(ego_state, ego_def))
    else
        # pull new frame from trajdata
        get!(simstate.scene, trajdata, simstate.frame)

        # move in propagated ego vehicle
        veh_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)
        simstate.scene[veh_index].state = ego_state
    end

    # update record
    update!(simstate.rec, simstate.scene)

    simstate
end


function tick(simparams::SimParams, u::Vector{Float64}, batch_index::Int=1)
    step_forward!(simparams.simstates[batch_index], simparams, u)
    simparams
end
function step_forward(simparams::SimParams, U::Matrix{Float64})
    # note: actions are batch_size × action_size
    for (i,s) in enumerate(simparams.simstates)
        step_forward!(s, simparams, U[i,:])
    end
    simparams
end

function reward(simstate::SimState, simparams::SimParams)

    veh_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)
    veh_ego = simstate.scene[veh_index]

    reward = 0.0
    if !simparams.use_debug_reward
        trajdata = simparams.trajdatas[simstate.trajdata_index]
        acc_ego = convert(Float64, get(ACC, simstate.rec, trajdata.roadway, veh_index))
        jrk_ego = convert(Float64, get(JERK, simstate.rec, trajdata.roadway, veh_index))
        ome_ego = convert(Float64, get(ANGULARRATEG, simstate.rec, trajdata.roadway, veh_index))
        cen_ego = veh_ego.state.posF.t

        d_ml = convert(Float64, get(MARKERDIST_LEFT, simstate.rec, trajdata.roadway, veh_index))
        d_mr = convert(Float64, get(MARKERDIST_RIGHT, simstate.rec, trajdata.roadway, veh_index))

        reward += simparams.col_weight * get_first_collision(simstate.scene, veh_index).is_colliding
        reward += simparams.off_weight * (d_ml < -1.0 || d_mr < -1.0)
        reward += simparams.rev_weight * (veh_ego.state.v < 0.0)
        reward += simparams.jrk_weight * jrk_ego*jrk_ego
        reward += simparams.acc_weight * acc_ego*acc_ego
        reward += simparams.cen_weight * cen_ego*cen_ego
        reward += simparams.ome_weight * ome_ego*ome_ego
    else
        reward -= (veh_ego.state.v^2)
        reward -= ((veh_ego.state.posG.θ - π/2)^2)
    end

    reward
end
function reward(simparams::SimParams, u::Vector{Float64}, batch_index::Int = 1)
    reward(simparams.simstates[batch_index], simparams)
end

function observe(simparams::SimParams, batch_index::Int=1)
    simstate = simparams.simstates[batch_index]
    trajdata = simparams.trajdatas[simstate.trajdata_index]
    veh_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)
    pull_features!(simparams.extractor, simparams.features, simstate.rec, trajdata.roadway, veh_index)
end

isdone(simparams::SimParams) = simparams.step_counter ≥ simparams.nsteps

function Base.step(simparams::SimParams, u::Vector{Float64}, batch_index::Int=1)

    r = reward(simparams, u, batch_index)
    tick(simparams, u, batch_index)
    features = observe(simparams, batch_index)
    simparams.step_counter += 1
    done = isdone(simparams)
    simstate = simparams.simstates[batch_index]

    # End if collision or reverse or off-road
    veh_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)
    trajdata = simparams.trajdatas[simstate.trajdata_index]
    d_ml = convert(Float64, get(MARKERDIST_LEFT, simstate.rec, trajdata.roadway, veh_index))
    d_mr = convert(Float64, get(MARKERDIST_RIGHT, simstate.rec, trajdata.roadway, veh_index))
    done = done && get_first_collision(simstate.scene, veh_index).is_colliding
    done = done && (d_ml < -1.0 || d_mr < -1.0)
    done = done && (simstate.scene[veh_index] < 0.0)

    (features, r, done)
end
function Base.step(simparams::SimParams, U::Matrix{Float64})

    batch_size = length(simparams.states)
    feature_mat = Array(Float64, batch_size, obssize(simparams))
    rewards = Array(Float64, batch_size)
    dones = Array(Float64, batch_size)

    step_counter = simparams.step_counter
    for batch_index in 1 : batch_size
        simparams.step_counter = step_counter
        features, reward, done = step(simparams, U[batch_index, :], batch_index)
        feature_mat[batch_index, :] = features
        rewards[batch_index] = reward
        dones[batch_index] = done
    end
    simparams.step_counter = step_counter + 1

    (feature_mat, rewards, dones)
end

action_space_bounds(simparams::SimParams) = ([-5.0, -1.0], [3.0, 1.0])
observation_space_bounds(simparams::SimParams) = (fill(-Inf, length(simparams.extractor)), fill(Inf, length(simparams.extractor)))

##################################
# Visualization

function AutoViz.render(simparams::SimParams, batch_index::Int=1)

    simstate = simparams.simstates[batch_index]
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    carcolors = Dict{Int,Colorant}()
    carcolors[simstate.egoid] = COLOR_CAR_EGO

    for id in simstate.playback_reactive_active_vehicle_ids
        carcolors[id] = colorant"0xF9971FFF" # orange
    end

    render(simstate.scene, trajdata.roadway,
          cam=CarFollowCamera(simstate.egoid, 5.0), car_colors=carcolors)
end

function AutoViz.render(simparams::SimParams, image::Matrix{UInt32}, batch_index::Int=1)

    simstate = simparams.simstates[batch_index]
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    c = CairoImageSurface(image, Cairo.FORMAT_ARGB32, flipxy=false)
    ctx = CairoContext(c)

    carcolors = Dict{Int,Colorant}()
    carcolors[simstate.egoid] = COLOR_CAR_EGO

    for id in simstate.playback_reactive_active_vehicle_ids
        carcolors[id] = colorant"0xF9971FFF" # orange
    end

    #render(ctx, simstate.scene, trajdata.roadway,
    #      cam=CarFollowCamera(simstate.egoid, 5.0), car_colors=carcolors)

    render(ctx, simstate.scene, trajdata.roadway, cam=CarFollowCamera(simstate.egoid, 5.0))

    image
end

function reel_drive(
    gif_filename::AbstractString,
    actions::Matrix{Float64}, # columns are actions
    simparams::SimParams
    )

    frames = Reel.Frames(MIME("image/png"), fps=framerate)

    action = [NaN,NaN]
    push!(frames, render(simparams))
    for frame_index in 1:size(actions,2)

        action[1] = actions[frame_index,1]
        action[2] = actions[frame_index,2]

        step_forward(simparams, action)

        push!(frames, render(simparams))
    end

    Reel.write(gif_filename, frames) # Write to a gif file
end

end # module
