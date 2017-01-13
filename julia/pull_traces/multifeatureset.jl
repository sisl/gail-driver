include("lidar_sensors.jl")

type MultiFeatureExtractor <: AbstractFeatureExtractor
    extract_core::Bool # [8] if true, extract lane offset, lane relative heading, speed, vehicle width, vehicle length, marker dist left+right, lane curvature
    extract_temporal::Bool # [6]  if true, extract accel, jerk, turnrate G, angular rate G, turnrate F, angular rate F
    extract_well_behaved::Bool # [3] if true, extract is_colliding, is_offroad, is_reversing
    extract_neighbor_features::Bool # [28] if true, extract everything that can be missing in tim2d features (ie, speed/dist fore, etc.)
    extract_carlidar_rangerate::Bool # [nbeams] if true, extract range rate as well as range

    carlidar::LidarSensor # [2*nbeams] NOTE: range rate can be turned off.
    roadlidar::RoadlineLidarSensor # [nbeams × depth]
    road_lidar_culling::RoadwayLidarCulling
end
function MultiFeatureExtractor(
    extract_core::Bool,
    extract_temporal::Bool,
    extract_well_behaved::Bool,
    extract_neighbor_features::Bool,
    extract_carlidar_rangerate::Bool,
    carlidar_nbeams::Int,
    roadlidar_nbeams::Int,
    roadlidar_nlanes::Int,
    ;
    carlidar_max_range::Float64 = 100.0, # [m]
    roadlidar_max_range::Float64 = 100.0, # [m]
    )

    carlidar = LidarSensor(carlidar_nbeams, max_range=carlidar_max_range, angle_offset=-π)
    roadlidar = RoadlineLidarSensor(roadlidar_nbeams, max_range=roadlidar_max_range, angle_offset=-π, max_depth=roadlidar_nlanes)

    MultiFeatureExtractor(
        extract_core,
        extract_temporal,
        extract_well_behaved,
        extract_neighbor_features,
        extract_carlidar_rangerate,
        carlidar,
        roadlidar,
        RoadwayLidarCulling(),
        )
end

AutomotiveDrivingModels.rec_length(ext::MultiFeatureExtractor) = ext.extract_temporal ? 3 : 1
function Base.length(ext::MultiFeatureExtractor)
    nbeams_carlidar = nbeams(ext.carlidar)
    nbeams_roadlidar = nbeams(ext.roadlidar)
    nlanes_roadlidar = nlanes(ext.roadlidar)

    8 * ext.extract_core +
    6 * ext.extract_temporal +
    3 * ext.extract_well_behaved +
    28 * ext.extract_neighbor_features +
    nbeams_carlidar * (1 + ext.extract_carlidar_rangerate) +
    nbeams_roadlidar * nlanes_roadlidar
end
function get_start_of_feature_section(ext::MultiFeatureExtractor)

    nbeams_carlidar = nbeams(ext.carlidar)
    nbeams_roadlidar = nbeams(ext.roadlidar)
    nlanes_roadlidar = nlanes(ext.roadlidar)

    i = 0
    core = temporal = well_behaved = neighbor = carlidar_range = carlidar_range_rate = roadlidar_range = -1

    if ext.extract_core
        core = 1
        i += 8
    end
    if ext.extract_temporal
        temporal = i
        i += 6
    end
    if ext.extract_well_behaved
        well_behaved = i
        i += 3
    end
    if ext.extract_neighbor_features
        neighbor = i
        i += 28
    end
    if nbeams_carlidar > 0
        carlidar_range = i
        i += nbeams_carlidar

        if ext.extract_carlidar_rangerate
            carlidar_range_rate = i
            i += nbeams_carlidar
        end
    end
    if nbeams_roadlidar > 0
        roadlidar_range = i
        i += nbeams_roadlidar * nlanes_roadlidar
    end

    (core, temporal, well_behaved, neighbor, carlidar_range, carlidar_range_rate, roadlidar_range, i)
end
function AutomotiveDrivingModels.pull_features!{F<:AbstractFloat}(ext::MultiFeatureExtractor,
    features::Vector{F},
    rec::SceneRecord,
    roadway::Roadway,
    vehicle_index::Int,
    pastframe::Int=0,
    )

    feature_index = 0
    scene = get_scene(rec, pastframe)
    veh_ego = scene[vehicle_index]

    d_ml = convert(Float64, get(MARKERDIST_LEFT, rec, roadway, vehicle_index, pastframe))
    d_mr = convert(Float64, get(MARKERDIST_RIGHT, rec, roadway, vehicle_index, pastframe))

    if ext.extract_core
        #=
        1 - lane offset (positive to left) [m]
        2 - lane relative heading (positive to left) [rad]
        3 - speed [m/s]
        4 - vehicle length [m]
        5 - vehicle width [m]
        6 - lane curvature [1/m]
        7 - Marker Dist left
        8 - Marker Dist right
        =#

        features[feature_index+=1] = veh_ego.state.posF.t
        features[feature_index+=1] = veh_ego.state.posF.ϕ
        features[feature_index+=1] = veh_ego.state.v
        features[feature_index+=1] = veh_ego.def.length
        features[feature_index+=1] = veh_ego.def.width
        features[feature_index+=1] = convert(Float64, get(LANECURVATURE, rec, roadway, vehicle_index, pastframe))
        features[feature_index+=1] = d_ml
        features[feature_index+=1] = d_mr
    end

    if ext.extract_temporal
        #=
        1 - previous acceleration [m/s²]
        2 - previous jerk [m/s³]
        3 - previous turnrate G [rad/s]
        4 - previous angular rate G [rad/s²]
        5 - previous turnrate F [rad/s]
        6 - previous angular rate F [rad/s²]
        =#

        features[feature_index+=1] = convert(Float64, get(ACC, rec, roadway, vehicle_index, pastframe))
        features[feature_index+=1] = convert(Float64, get(JERK, rec, roadway, vehicle_index, pastframe))
        features[feature_index+=1] = convert(Float64, get(TURNRATEG, rec, roadway, vehicle_index, pastframe))
        features[feature_index+=1] = convert(Float64, get(ANGULARRATEG, rec, roadway, vehicle_index, pastframe))
        features[feature_index+=1] = convert(Float64, get(TURNRATEF, rec, roadway, vehicle_index, pastframe))
        features[feature_index+=1] = convert(Float64, get(ANGULARRATEF, rec, roadway, vehicle_index, pastframe))
    end

    if ext.extract_well_behaved
        #=
        1 - is colliding
        2 - is offroad
        3 - is reversing
        =#

        features[feature_index+=1] = convert(Float64, get(IS_COLLIDING, rec, roadway, vehicle_index, pastframe))
        features[feature_index+=1] = convert(Float64, d_ml < -1.0 || d_mr < -1.0)
        features[feature_index+=1] = convert(Float64, veh_ego.state.v < 0.0)
    end

    if ext.extract_neighbor_features
        #=
        can be missing: - first is feature, 2nd is indicator whether it is missing (0 if exists, 1 if missing)
        (1, 2) - lane offset from left lane
        (3, 4) - lane offset from right lane
        (5, 6) - speed fore middle
        (7, 8) - dist fore middle (distance along the lane from the front of the ego car to the rear of the ego car in the same lane)
        (9, 10) - speed fore left
        (11,12) - dist fore left
        (13,14) - speed fore right
        (15,16) - dist fore right
        (17,18) - speed rear middle
        (19,20) - dist fore middle
        (21,22) - speed rear left
        (23,24) - dist rear left
        (25,26) - speed rear right
        (27,28) - dist rear right
        =#

        vtpf = VehicleTargetPointFront()
        vtpr = VehicleTargetPointRear()
        fore_M = get_neighbor_fore_along_lane(      scene, vehicle_index, roadway, vtpf, vtpr, vtpf)
        fore_L = get_neighbor_fore_along_left_lane( scene, vehicle_index, roadway, vtpf, vtpr, vtpf)
        fore_R = get_neighbor_fore_along_right_lane(scene, vehicle_index, roadway, vtpf, vtpr, vtpf)
        rear_M = get_neighbor_rear_along_lane(      scene, vehicle_index, roadway, vtpr, vtpf, vtpr)
        rear_L = get_neighbor_rear_along_left_lane( scene, vehicle_index, roadway, vtpr, vtpf, vtpr)
        rear_R = get_neighbor_rear_along_right_lane(scene, vehicle_index, roadway, vtpr, vtpf, vtpr)

        function set_feature_missing!(features::Vector, i::Int)
            features[i] = 0.0
            features[i+1] = 1.0
            features
        end
        function set_feature!(features::Vector, i::Int, v::Float64)
            features[i] = v
            features[i+1] = 0.0
            features
        end
        function set_dual_feature!(features::Vector, i::Int, f::FeatureValue)
            if f.i == FeatureState.MISSING
                set_feature_missing!(features, i)
            else
                set_feature!(features, i, f.v)
            end
            features
        end
        function set_speed_and_distance!(features::Vector, i::Int, neigh::NeighborLongitudinalResult)
            neigh.ind != 0 ? set_feature!(features, i, scene[neigh.ind].state.v) :
                              set_feature_missing!(features, i)
            neigh.ind != 0 ? set_feature!(features, i+2, neigh.Δs) :
                              set_feature_missing!(features, i+2)
            features
        end

        set_dual_feature!(features, feature_index+=1, get(LANEOFFSETLEFT, rec, roadway, vehicle_index, pastframe))
        feature_index+=1
        set_dual_feature!(features, feature_index+=1, get(LANEOFFSETRIGHT, rec, roadway, vehicle_index, pastframe))
        feature_index+=1

        set_speed_and_distance!(features, feature_index+=1, fore_M); feature_index+=3
        set_speed_and_distance!(features, feature_index+=1, fore_L); feature_index+=3
        set_speed_and_distance!(features, feature_index+=1, fore_R); feature_index+=3
        set_speed_and_distance!(features, feature_index+=1, rear_M); feature_index+=3
        set_speed_and_distance!(features, feature_index+=1, rear_L); feature_index+=3
        set_speed_and_distance!(features, feature_index+=1, rear_R); feature_index+=3
    end

    nbeams_carlidar = nbeams(ext.carlidar)
    if nbeams_carlidar > 0
        observe!(ext.carlidar, scene, roadway, vehicle_index)
        copy!(features, feature_index+=1, ext.carlidar.ranges, 1)
        feature_index += nbeams_carlidar - 1
        if ext.extract_carlidar_rangerate
            copy!(features, feature_index+=1, ext.carlidar.range_rates, 1)
            feature_index += nbeams_carlidar - 1
        end
    end

    nbeams_roadlidar = nbeams(ext.roadlidar)
    if nbeams_roadlidar > 0
        if ext.road_lidar_culling.is_leaf
            observe!(ext.roadlidar, scene, roadway, vehicle_index)
        else
            observe!(ext.roadlidar, scene, roadway, vehicle_index, ext.road_lidar_culling)
        end
        copy!(features, feature_index+=1, ext.roadlidar.ranges)
        feature_index += length(ext.roadlidar.ranges) - 1
    end

    @assert(feature_index == length(features))
    if findfirst(v->isnan(v), features) != 0
        error("feature $(findfirst(v->isnan(v), features)) is nan")
    end

    features
end

