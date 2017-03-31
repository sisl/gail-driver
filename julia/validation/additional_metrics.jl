using AutomotiveDrivingModels
using AutoDrivers.GaussianMixtureRegressionDrivers

########################################
#            CollisionRate             #
########################################

type CollisionRate <: TraceMetricExtractor
    running_sum::Float64
    n_obs::Int

    CollisionRate(running_sum::Float64=0.0, n_obs::Int=0) = new(running_sum, n_obs)
end
Base.Symbol(::CollisionRate) = :collisionrate
get_score(m::CollisionRate) = m.running_sum / m.n_obs
function reset!(metric::CollisionRate)
    metric.running_sum = 0.0
    metric.n_obs = 0
    metric
end
function extract_collision_rate(rec::SceneRecord, roadway::Roadway, egoid::Int)
    collided = 0.0
    for pastframe in 1 - length(rec) : 0
        vehicle_index = get_index_of_first_vehicle_with_id(rec, egoid, pastframe)
        collide = convert(Float64, get(IS_COLLIDING, rec, roadway, vehicle_index, pastframe))
        collided = min(collide + collided, 1.0) # only count if first collision
    end
    collided
end
function extract!(
    metric::CollisionRate,
    rec_orig::SceneRecord, # the records are exactly as long as the simulation (ie, contain everything)
    rec_sim::SceneRecord,
    roadway::Roadway,
    egoid::Int;
    real::Bool=false,
    )
    if real
        metric.running_sum += extract_collision_rate(rec_orig, roadway, egoid)
    else
        metric.running_sum += extract_collision_rate(rec_sim, roadway, egoid)
    end
    metric.n_obs += 1

    metric
end

########################################
#            OffRoadRate               #
########################################

type OffRoadRate <: TraceMetricExtractor
    running_sum::Float64
    n_obs::Int

    OffRoadRate(running_sum::Float64=0.0, n_obs::Int=0) = new(running_sum, n_obs)
end
Base.Symbol(::OffRoadRate) = :offroadrate
get_score(m::OffRoadRate) = m.running_sum / m.n_obs
function reset!(metric::OffRoadRate)
    metric.running_sum = 0.0
    metric.n_obs = 0
    metric
end
function extract_off_road_rate(rec::SceneRecord, roadway::Roadway, egoid::Int)
    num_off_road = 0.0
    for pastframe in 1 - length(rec) : 0
        vehicle_index = get_index_of_first_vehicle_with_id(rec, egoid, pastframe)
        off_road = (convert(Float64, get(MARKERDIST_LEFT, rec, roadway, vehicle_index, pastframe)) < -1.0) +
                    (convert(Float64, get(MARKERDIST_RIGHT, rec, roadway, vehicle_index, pastframe)) < -1.0)
        num_off_road += off_road
    end
    num_off_road
end
function extract!(
    metric::OffRoadRate,
    rec_orig::SceneRecord, # the records are exactly as long as the simulation (ie, contain everything)
    rec_sim::SceneRecord,
    roadway::Roadway,
    egoid::Int;
    real::Bool=false,
    )
    if real
        metric.running_sum += extract_off_road_rate(rec_orig, roadway, egoid)
    else
        metric.running_sum += extract_off_road_rate(rec_sim, roadway, egoid)
    end
    metric.n_obs += 1

    metric
end

########################################
#            LaneChangeRate            #
########################################

type LaneChangeRate <: TraceMetricExtractor
    running_sum::Float64
    n_obs::Int

    LaneChangeRate(running_sum::Float64=0.0, n_obs::Int=0) = new(running_sum, n_obs)
end
Base.Symbol(::LaneChangeRate) = :lanechangerate
get_score(m::LaneChangeRate) = m.running_sum / m.n_obs
function reset!(metric::LaneChangeRate)
    metric.running_sum = 0.0
    metric.n_obs = 0
    metric
end
function extract_lane_change_rate(rec::SceneRecord, roadway::Roadway, egoid::Int)
    num_lane_change = 0.0
    vehicle_index = get_index_of_first_vehicle_with_id(rec, egoid, 1 - length(rec))
    veh_ego = rec[vehicle_index, 1 - length(rec)]
    prev_lane = roadway[veh_ego.state.posF.roadind.tag] # Find initial lane
    for pastframe in 2 - length(rec) : 0
        vehicle_index = get_index_of_first_vehicle_with_id(rec, egoid, pastframe)
        veh_ego = rec[vehicle_index, pastframe]
        lane = roadway[veh_ego.state.posF.roadind.tag]
        num_lane_change += (lane != prev_lane)
        prev_lane = lane
    end
    num_lane_change
end
function extract!(
    metric::LaneChangeRate,
    rec_orig::SceneRecord, # the records are exactly as long as the simulation (ie, contain everything)
    rec_sim::SceneRecord,
    roadway::Roadway,
    egoid::Int;
    real::Bool=false,
    )
    if real
        metric.running_sum += extract_lane_change_rate(rec_orig, roadway, egoid)
    else
        metric.running_sum += extract_lane_change_rate(rec_sim, roadway, egoid)
    end
    metric.n_obs += 1

    metric
end

########################################
#            HardBrakeRate             #
########################################

type HardBrakeRate <: TraceMetricExtractor
    running_sum::Float64
    n_obs::Int

    HardBrakeRate(running_sum::Float64=0.0, n_obs::Int=0) = new(running_sum, n_obs)
end
Base.Symbol(::HardBrakeRate) = :hardbrakerate
get_score(m::HardBrakeRate) = m.running_sum / m.n_obs
function reset!(metric::HardBrakeRate)
    metric.running_sum = 0.0
    metric.n_obs = 0
    metric
end
function extract_hard_brake_rate(rec::SceneRecord, roadway::Roadway, egoid::Int)
    num_hard_brake = 0.0
    for pastframe in 2 - length(rec) : 0
        vehicle_index = get_index_of_first_vehicle_with_id(rec, egoid, pastframe)
        acc_ego = convert(Float64, get(ACC, rec, roadway, vehicle_index, pastframe))
        num_hard_brake += (acc_ego < -3.0)
    end
    num_hard_brake
end
function extract!(
    metric::HardBrakeRate,
    rec_orig::SceneRecord, # the records are exactly as long as the simulation (ie, contain everything)
    rec_sim::SceneRecord,
    roadway::Roadway,
    egoid::Int;
    real::Bool=false,
    )
    if real
        metric.running_sum += extract_hard_brake_rate(rec_orig, roadway, egoid)
    else
        metric.running_sum += extract_hard_brake_rate(rec_sim, roadway, egoid)
    end
    metric.n_obs += 1

    metric
end

########################################
#        RWSE Global Position          #
########################################

type RWSEPosG <: TraceMetricExtractor
    horizon::Float64 # [s]
    running_sum::Float64
    n_obs::Int
end
RWSEPosG(horizon::Float64) = RWSEPosG(horizon, 0.0, 0)

Base.Symbol(m::RWSEPosG) = Symbol(@sprintf("RWSE_posG_%d_%02d", floor(Int, m.horizon), floor(Int, 100*rem(m.horizon, 1.0))))
get_score(m::RWSEPosG) = sqrt(m.running_sum / m.n_obs)
function reset!(metric::RWSEPosG)
    metric.running_sum = 0.0
    metric.n_obs = 0
    metric
end
function extract!(
    metric::RWSEPosG,
    rec_orig::SceneRecord, # the records are exactly as long as the simulation (ie, contain everything)
    rec_sim::SceneRecord,
    roadway::Roadway,
    egoid::Int,
    )

    # TODO: how to handle missing values???

    pastframe = 1-length(rec_orig) + clamp(round(Int, metric.horizon/rec_orig.timestep), 0, length(rec_orig)-1)

    # pull true value
    vehicle_index = get_index_of_first_vehicle_with_id(rec_orig, egoid, pastframe)
    veh_ego = rec_orig[vehicle_index, pastframe]
    x_true = veh_ego.state.posG.x
    y_true = veh_ego.state.posG.y

    # pull sim value
    vehicle_index = get_index_of_first_vehicle_with_id(rec_sim, egoid, pastframe)
    veh_ego = rec_sim[vehicle_index, pastframe]
    x_montecarlo = veh_ego.state.posG.x
    y_montecarlo = veh_ego.state.posG.y

    d = (x_true - x_montecarlo)^2 + (y_true - y_montecarlo)^2
    
    metric.running_sum += d
    metric.n_obs += 1

    metric
end

# Check for outliers, which seem to  occur with controller rollouts
function check_outliers(rec_orig::SceneRecord, rec_sim::SceneRecord, egoid::Int)
    pastframe = 1-length(rec_orig) + clamp(round(Int, 5.0/rec_orig.timestep), 0, length(rec_orig)-1)

    # pull true value
    vehicle_index = get_index_of_first_vehicle_with_id(rec_orig, egoid, pastframe)
    veh_ego = rec_orig[vehicle_index, pastframe]
    x_true = veh_ego.state.posG.x
    y_true = veh_ego.state.posG.y

    # pull sim value
    vehicle_index = get_index_of_first_vehicle_with_id(rec_sim, egoid, pastframe)
    veh_ego = rec_sim[vehicle_index, pastframe]
    x_montecarlo = veh_ego.state.posG.x
    y_montecarlo = veh_ego.state.posG.y

    d = sqrt((x_true - x_montecarlo)^2 + (y_true - y_montecarlo)^2)
    return (d >= 50.0)
end

function rollout!(
    rec::SceneRecord,
    model::DriverModel,
    egoid::Int,
    trajdata::Trajdata,
    time_start::Float64,
    time_end::Float64,
    simparams::Auto2D.SimParams;
    prime_history::Int = 0,
    )
    
    # Initialize values
    Δt = rec.timestep
    simstate = simparams.simstates[1]

    # clear rec and make first observations
    if Symbol("gru") in fieldnames(model.net)
        model.net[:gru].h_prev = zeros(length(model.net[:gru].h_prev))
    end
    empty!(simstate.rec)
    empty!(rec)
    reset_hidden_state!(model)
    for i in 1 : prime_history
        t = time_start - Δt * (prime_history - i + 1)
        update!(simstate.rec, get!(simstate.scene, trajdata, t))
        AutomotiveDrivingModels.observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)
    end

    # empty playback reactive
    empty!(simstate.playback_reactive_active_vehicle_ids)

    # Run start time step
    t = time_start
    update!(simstate.rec, get!(simstate.scene, trajdata, t))
    update!(model.rec, simstate.scene)
    AutomotiveDrivingModels.observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)

    # run simulation
    while t < time_end

        # Find action and step forward
        
        ego_action = rand(model)
        a = clamp(ego_action.a, -5.0, 3.0)
        ω = clamp(ego_action.ω, -0.1, 0.1)
        Auto2D.step(simparams, [a, ω])

        # update record
        AutomotiveDrivingModels.observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)
        update!(rec, simstate.scene)

        # update time
        t += Δt
    end
    rec
end

function simulate!(
    rec::SceneRecord,
    model::DriverModel,
    egoid::Int,
    trajdata::Trajdata,
    time_start::Float64,
    time_end::Float64,
    simparams::Auto2D.SimParams;
    prime_history::Int = 0,
    )
    
    # Initialize values
    Δt = rec.timestep
    simstate = simparams.simstates[1]

    # clear rec and make first observations
    empty!(simstate.rec)
    empty!(rec)
    reset_hidden_state!(model)
    for i in 1 : prime_history
        t = time_start - Δt * (prime_history - i + 1)
        update!(simstate.rec, get!(simstate.scene, trajdata, t))
        AutomotiveDrivingModels.observe!(model, simstate.scene, trajdata.roadway, egoid)
    end

    # empty playback reactive
    empty!(simstate.playback_reactive_active_vehicle_ids)

    # Run start time step
    t = time_start
    update!(simstate.rec, get!(simstate.scene, trajdata, t))

    # If controller, set desired speed to be inital speed
    veh_index = get_index_of_first_vehicle_with_id(simstate.scene, egoid)
    if simparams.ego_action_type == LatLonAccel
        set_desired_speed!(model, simstate.scene[veh_index].state.v)
    end
    AutomotiveDrivingModels.observe!(model, simstate.scene, trajdata.roadway, egoid)

    # run simulation
    while t < time_end

        # Find action and step forward
        ego_action = rand(model)
        veh_index = get_index_of_first_vehicle_with_id(simstate.scene, egoid)
        if simstate.scene[veh_index].state.v < 0.0 && (simparams.ego_action_type == LatLonAccel)
            ego_action = LatLonAccel(ego_action.a_lat, max(ego_action.a_lon, 0.0))
        end
        if simparams.ego_action_type == AccelTurnrate
            Auto2D.step(simparams, [ego_action.a, ego_action.ω])
        else
            Auto2D.step(simparams, [ego_action.a_lat, a_lon])
        end
        AutomotiveDrivingModels.observe!(model, simstate.scene, trajdata.roadway, egoid)

        # update record
        update!(rec, simstate.scene)

        # update time
        t += Δt
    end
    rec
end

# Initialize values for simstate
function reset_simstate!(simstate::Auto2D.SimState, seg::TrajdataSegment)
    simstate.egoid = seg.egoid
    simstate.trajdata_index = seg.trajdata_index
    simstate.frame = seg.frame_lo
end

########################################
#        Calculate Metric Scores       #
########################################

function calc_metrics!(
    metrics_df::DataFrame,
    model::DriverModel,
    metrics::Vector{TraceMetricExtractor},
    simparams::Auto2D.SimParams,
    foldset_seg_test::FoldSet; # should match the test segments in evaldata
    n_simulations_per_trace::Int = 10,
    row::Int = foldset_seg_test.fold, # row in metrics_df to write to
    prime_history::Int = 0,
    calc_logl::Bool = true,
    )
    # reset metrics
    for metric in metrics
        try
            reset!(metric)
        catch
            AutomotiveDrivingModels.reset!(metric)
        end
    end

    logl = 0.0
    n_traces = 0

    # Set correct action type
    if Symbol("a") in fieldnames(rand(model))
        simparams.ego_action_type = AccelTurnrate
    else
        simparams.ego_action_type = LatLonAccel
    end

    # simulate traces and perform online metric extraction
    scene = Scene()
    for seg_index in foldset_seg_test
        seg = simparams.segments[seg_index]
        trajdata = simparams.trajdatas[seg.trajdata_index]

        rec_orig = pull_record(seg, trajdata, prime_history) # TODO - make efficient
        rec_sim = deepcopy(rec_orig)

        time_start = get_time(trajdata, seg.frame_lo)
        time_end = get_time(trajdata, seg.frame_hi)
        n_traces += 1

        for sim_index in 1 : n_simulations_per_trace
            reset_simstate!(simparams.simstates[1], seg)
            
            if Symbol("net") in fieldnames(model)
                rollout!(rec_sim, model, seg.egoid, trajdata,
                          time_start, time_end, simparams, prime_history=prime_history)      
            else
                simulate!(rec_sim, model, seg.egoid, trajdata,
                          time_start, time_end, simparams, prime_history=prime_history)
            end

            for metric in metrics
                if !check_outliers(rec_orig, rec_sim, seg.egoid)
                    try
                        extract!(metric, rec_orig, rec_sim, trajdata.roadway, seg.egoid)
                    catch
                        AutomotiveDrivingModels.extract!(metric, rec_orig, rec_sim, trajdata.roadway, seg.egoid)
                    end
                end
            end
        end
    end

    # compute metric scores
    for metric in metrics
        try
            metrics_df[row, Symbol(metric)] = get_score(metric)
        catch
            metrics_df[row, Symbol(metric)] = AutomotiveDrivingModels.get_score(metric)
        end
    end
    metrics_df[row, :time] = string(now())
    metrics_df[row, :logl] = logl/n_traces

    metrics_df
end
