module Auto1D

using AutomotiveDrivingModels
using AutoViz
import Reel

export gen_simparams, reset, tick, reward, observe, step, isdone, action_space_bounds, observation_space_bounds
export SimParams, reel_drive

type SimState

    scene::Scene

    function SimState(roadway::Roadway)
        retval = new()
        retval.scene = Scene()

        push!(retval.scene,Vehicle(VehicleState(VecSE2(7.0 + rand()*45.0,randn()*0.5,0.0), roadway, 21.0),
                    VehicleDef(1, AgentClass.CAR, 4.826, 1.81)))
        push!(retval.scene,Vehicle(VehicleState(VecSE2(0.,randn()*0.5,0.0), roadway, 15.0 + rand()*10.0),
                        VehicleDef(2, AgentClass.CAR, 4.826, 1.81)))
        retval
    end
end
type SimParams
    col_weight::Float64 # reward for collision
    rev_weight::Float64 # reward for reversal (negative speeds)
    jrk_weight::Float64 # reward scale for square jerk
    acc_weight::Float64 # reward scale for square acceleration

    rec::SceneRecord
    model_rear::LaneFollowingDriver
    model_fore::LaneFollowingDriver
    nsteps::Int
    step_counter::Int
    simstates::Vector{SimState}
    roadway::Roadway
end

function gen_simparams(batch_size::Int, args::Dict)

    col_weight = get(args, "col_weight", -2.0)
    rev_weight = get(args, "rev_weight", -0.50)
    jrk_weight = get(args, "jrk_weight", -0.050)
    acc_weight = get(args, "acc_weight", -0.050)
    nsteps = get(args, "nsteps", 100)
    timestep = get(args, "timestep", 0.1)

    roadway = gen_straight_roadway(1, 10000.0)
    simstates = Array(SimState, batch_size)
    for i in 1 : batch_size
        simstates[i] = SimState(roadway)
    end

    rec = SceneRecord(1, timestep)
    context = IntegratedContinuous(timestep, 1)
    model_rear = LaneFollowingDriver(context, StaticLongitudinalDriver())
    model_fore = LaneFollowingDriver(context, IntelligentDriverModel(σ=0.2, v_des=21.0))

    SimParams(col_weight, rev_weight, jrk_weight, acc_weight,
              rec, model_rear, model_fore,
              nsteps, 0, simstates, roadway)
end
Base.show(io::IO, simparams::SimParams) = print(io, "SimParams")

function restart!(simstate::SimState, simparams::SimParams)

    roadway = simparams.roadway
    simstate.scene[1].state = VehicleState(VecSE2(7.0 + rand()*45.0,randn()*0.5,0.0), roadway, 21.0)
    simstate.scene[2].state = VehicleState(VecSE2(0.0, randn()*0.5,0.0), roadway, 15.0 + rand()*10.0)

    simstate
end
function Base.reset(simparams::SimParams)
    for state in simparams.simstates
        restart!(state, simparams)
    end
    reset_hidden_state!(simparams.model_fore)
    reset_hidden_state!(simparams.model_rear)
    simparams.step_counter = 0
    simparams
end

function step_forward!(simstate::SimState, simparams::SimParams, action_ego::Vector{Float64})

    observe!(simparams.model_fore, simstate.scene, simparams.roadway, 1)
    context = action_context(simparams.model_fore)
    simstate.scene[1].state = propagate(simstate.scene[1], rand(simparams.model_fore), context, simparams.roadway)

    observe!(simparams.model_rear, simstate.scene, simparams.roadway, 2)
    simparams.model_rear.mlon.a = action_ego[1]
    context = action_context(simparams.model_rear)
    simstate.scene[2].state = propagate(simstate.scene[2], rand(simparams.model_rear), context, simparams.roadway)

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

    veh_index = 2
    veh_ego = simstate.scene[veh_index]

    empty!(simparams.rec)
    update!(simparams.rec, simstate.scene)
    roadway = simparams.roadway
    acc_ego = convert(Float64, get(ACC, simparams.rec, roadway, veh_index))
    jrk_ego = convert(Float64, get(JERK, simparams.rec, roadway, veh_index))

    reward = 0.0
    reward += simparams.col_weight * get_first_collision(simstate.scene, veh_index).is_colliding
    reward += simparams.rev_weight * (veh_ego.state.v < 0.0)
    reward += simparams.jrk_weight * jrk_ego*jrk_ego
    reward += simparams.acc_weight * acc_ego*acc_ego

    reward
end
function reward(simparams::SimParams, u::Vector{Float64}, batch_index::Int = 1)
    reward(simparams.simstates[batch_index], simparams)
end

function observe(simparams::SimParams, batch_index::Int=1)
    simstate = simparams.simstates[batch_index]

    veh_index = 2
    veh_ego = simstate.scene[veh_index]

    F = VehicleTargetPointFront()
    R = VehicleTargetPointRear()
    foreinfo = get_neighbor_fore_along_lane(simstate.scene, veh_index, simparams.roadway, F, R, F)

    features = [0.0, 0.0, 0.0, 1.0]

    if foreinfo.ind != 0
        fore = simstate.scene[foreinfo.ind]
        features[1] = foreinfo.Δs - simstate.scene[1].def.length/2 - simstate.scene[2].def.length/2
        features[2] = fore.state.v - veh_ego.state.v
        features[3] = veh_ego.state.v
        features[4] = 1.0
    end

    features
end

isdone(simparams::SimParams) = simparams.step_counter ≥ simparams.nsteps

function Base.step(simparams::SimParams, u::Vector{Float64}, batch_index::Int=1)

    r = reward(simparams, u, batch_index)
    tick(simparams, u, batch_index)
    features = observe(simparams, batch_index)
    simparams.step_counter += 1
    done = isdone(simparams)

    (features, r, done)
end
function Base.step(simparams::SimParams, U::Matrix{Float64})

    batch_size = length(simparams.states)
    feature_mat = Array(Float64, batch_size, _obssize(simparams))
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

action_space_bounds(simparams::SimParams) = ([-5.0], [3.0])
observation_space_bounds(simparams::SimParams) = ([-Inf, -Inf, -Inf, 0.0], [Inf, Inf, Inf, 1.0])

function reel_drive_1d(
    gif_filename::AbstractString,
    veh2_actions::Vector{Float64}, # actions are for model2, just longitudinal
    scene::Scene, # initial scene, is overwritten
    model1::DriverModel{LaneFollowingAccel, IntegratedContinuous}, # IDM
    model2::DriverModel{LaneFollowingAccel, IntegratedContinuous}, # StaticLognitudinal
    roadway::Roadway;
    framerate::Int=10,
    overlays::Vector{SceneOverlay}=SceneOverlay[], # CarFollowingStatsOverlay(2)
    cam::Camera=FitToContentCamera(),
    )

    actions = Array(LaneFollowingAccel, length(scene))
    models = Dict{Int, DriverModel}()
    models[1] = model1
    models[2] = model2

    frames = Reel.Frames(MIME("image/png"), fps=framerate)

    push!(frames, render(scene, roadway, overlays, cam=cam))
    for frame_index in 1:length(veh2_actions)

        actions[1] = rand(observe!(model1, scene, roadway, 1))

        observe!(model2, scene, roadway, 1)
        model2.mlon.a = veh2_actions[frame_index]
        actions[2] = rand(model2)

        tick!(scene, roadway, actions, models)
        push!(frames, render(scene, roadway, overlays, cam=cam))
    end

    Reel.write(gif_filename, frames) # Write to a gif file
end

end # module
