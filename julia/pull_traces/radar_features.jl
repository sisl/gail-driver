using AutoViz
using Colors

#############################################################
# Radar Sensor

type RadarSensor
    angles::Vector{Float64}
    ranges::Vector{Float64}
    range_rates::Vector{Float64}
    max_range::Float64
    poly::ConvexPolygon
end
function RadarSensor(nbeams::Int;
    max_range::Float64=100.0,
    angle_offset::Float64=0.0,
    )
    angles = collect(linspace(angle_offset,2*pi+angle_offset,nbeams+1))[2:end]
    ranges = Array(Float64, nbeams)
    range_rates = Array(Float64, nbeams)
    RadarSensor(angles, ranges, range_rates, max_range, ConvexPolygon(4))
end
function observe!(radar::RadarSensor, scene::Scene, roadway::Roadway, vehicle_index::Int)
    state_ego = scene[vehicle_index].state
    egoid = scene[vehicle_index].def.id
    ego_vel = polar(state_ego.v, state_ego.posG.θ)
    for (i,angle) in enumerate(radar.angles)
        ray_angle = state_ego.posG.θ + angle
        ray_vec = polar(1.0, ray_angle)
        ray = VecSE2(state_ego.posG.x, state_ego.posG.y, ray_angle)

        range = radar.max_range
        range_rate = 0.0
        for veh in scene
            if veh.def.id != egoid
                to_oriented_bounding_box!(radar.poly, veh)
                range2 = AutomotiveDrivingModels.get_collision_time(ray, radar.poly, 1.0)
                if !isnan(range2) && range2 < range
                    range = range2
                    relative_speed = polar(veh.state.v, veh.state.posG.θ) - ego_vel
                    range_rate = proj(relative_speed, ray_vec, Float64)
                end
            end
        end
        radar.ranges[i] = range
        radar.range_rates[i] = range_rate
    end

    radar
end
function render_radar!(rendermodel::RenderModel, radar::RadarSensor, posG::VecSE2;
    color::Colorant=colorant"white",
    line_width::Float64 = 0.05,
    )

    for (angle, range, range_rate) in zip(radar.angles, radar.ranges, radar.range_rates)
        ray = VecSE2(posG.x, posG.y, posG.θ + angle)
        t = 2 / (1 + exp(-range_rate))
        ray_color = t > 1.0 ? lerp(RGBA(1.0,1.0,1.0,0.5), RGBA(0.2,0.2,1.0,0.5), t/2) :
                              lerp(RGBA(1.0,0.2,0.2,0.5), RGBA(1.0,1.0,1.0,0.5), t)
        render_ray!(rendermodel::RenderModel, ray, color=ray_color, length=range, line_width=line_width)
    end
    rendermodel
end

#############################################################
# Road Line Radar Sensor

type RoadlineRadarSensor
    angles::Vector{Float64}
    ranges::Matrix{Float64} # [n_lanes by nbeams]
    max_range::Float64
    poly::ConvexPolygon
end
function RoadlineRadarSensor(nbeams::Int;
    max_range::Float64=100.0,
    max_depth::Int=2,
    angle_offset::Float64=0.0,
    )

    angles = collect(linspace(angle_offset,2*pi+angle_offset,nbeams+1))[2:end]
    ranges = Array(Float64, max_depth, nbeams)
    RoadlineRadarSensor(angles, ranges, max_range, ConvexPolygon(4))
end
function _update_radar!(radar::RoadlineRadarSensor, ray, p_lo, p_hi, beam_index)
    test_range = get_collision_time(ray, AutomotiveDrivingModels.LineSegment(p_lo, p_hi), 1.0)

    n_ranges = size(radar.ranges, 1)
    if !isnan(test_range)
        for k in 1 : n_ranges
            if test_range < radar.ranges[k,beam_index]
                for l in k+1 : n_ranges
                    radar.ranges[l,beam_index] = radar.ranges[l-1,beam_index]
                end
                radar.ranges[k,beam_index] = test_range
                break
            end
        end
    end
    radar
end
function observe!(radar::RoadlineRadarSensor, scene::Scene, roadway::Roadway, vehicle_index::Int)
    state_ego = scene[vehicle_index].state
    egoid = scene[vehicle_index].def.id
    ego_vel = polar(state_ego.v, state_ego.posG.θ)

    fill!(radar.ranges, radar.max_range)
    for (i,angle) in enumerate(radar.angles)
        ray_angle = state_ego.posG.θ + angle
        ray_vec = polar(1.0, ray_angle)
        ray = VecSE2(state_ego.posG.x, state_ego.posG.y, ray_angle)

        for seg in roadway.segments
            for lane in seg.lanes

                N = length(lane.curve)
                halfwidth = lane.width/2

                # always render the left lane marking
                p_lo = convert(VecE2, lane.curve[1].pos + polar(halfwidth, lane.curve[1].pos.θ + π/2))
                for j in 2:length(lane.curve)
                    p_hi = convert(VecE2, lane.curve[j].pos + polar(halfwidth, lane.curve[j].pos.θ + π/2))
                    _update_radar!(radar, ray, p_lo, p_hi, i)
                    p_lo = p_hi
                end
                if has_next(lane)
                    lane2 = next_lane(lane, roadway)
                    pt = lane2.curve[1]
                    p_hi = convert(VecE2, pt.pos + polar(lane2.width/2, pt.pos.θ + π/2))
                    _update_radar!(radar, ray, p_lo, p_hi, i)
                end

                # only check the right lane marking if this is the first lane
                if lane.tag.lane == 1
                    p_lo = convert(VecE2, lane.curve[1].pos + polar(halfwidth, lane.curve[1].pos.θ - π/2))
                    for j in 2:length(lane.curve)
                        p_hi = convert(VecE2, lane.curve[j].pos + polar(halfwidth, lane.curve[j].pos.θ - π/2))
                        _update_radar!(radar, ray, p_lo, p_hi, i)
                        p_lo = p_hi
                    end
                    if has_next(lane)
                        lane2 = next_lane(lane, roadway)
                        pt = lane2.curve[1]
                        p_hi = convert(VecE2, pt.pos + polar(lane2.width/2, pt.pos.θ - π/2))
                        _update_radar!(radar, ray, p_lo, p_hi, i)
                    end
                end
            end
        end
    end

    radar
end
function render_radar!(rendermodel::RenderModel, radar::RoadlineRadarSensor, posG::VecSE2;
    color::Colorant=RGBA(1.0,1.0,1.0,0.5),
    line_width::Float64 = 0.05,
    depth_level::Int = 1, # if 1, render first lanes struck by beams, if 2 render 2nd ...
    )

    for (angle, range) in zip(radar.angles, radar.ranges[depth_level,:])
        ray = VecSE2(posG.x, posG.y, posG.θ + angle)
        render_ray!(rendermodel::RenderModel, ray, color=color, length=range, line_width=line_width)
    end
    rendermodel
end

#############################################################
# Feature Extraction

const RADAR = RadarSensor(200, max_range=100.0, angle_offset=-π) # car distances and range rates
const LINERADAR = RoadlineRadarSensor(200, max_range=100.0, angle_offset=-π, max_depth=3) # lane edges
const N_FEATURES = 2*length(RADAR.ranges) + length(LINERADAR.ranges)
function AutomotiveDrivingModels.pull_features!{Fl<:AbstractFloat}(features::Array{Fl}, rec::SceneRecord, roadway::Roadway, vehicle_index::Int, pastframe::Int=0;
    radar::RadarSensor = RADAR,
    lineradar::RoadlineRadarSensor = LINERADAR,
    )

    #=
    let N = length(radar.ranges)
    let M = size(lineradar.ranges, 2)
    let max_depth = size(lineradar.ranges, 1)

    1 : N is the radar ranges; distance to the struck object, maxes out
    N+1 : 2N is the radar range rates
    2N+1 : 2N + M is line radar ranges at depth 1
    2N+M+1 : 2N + 2M is line radar ranges at depth 2
    ...
    2N+(max_depth-1)×M+1 : 2N + max_depth×M is line radar ranges at depth max_depth
    =#

    scene = get_scene(rec, pastframe)

    observe!(radar, scene, roadway, vehicle_index)
    observe!(lineradar, scene, roadway, vehicle_index)

    N = length(radar.ranges)
    M = size(lineradar.ranges, 2)
    max_depth = size(lineradar.ranges, 1)

    copy!(features, 1, radar.ranges, 1)
    copy!(features, N+1, radar.range_rates, 1)
    copy!(features, 2N+1, lineradar.ranges, 1)

    if findfirst(v->isnan(v), features) != 0
        error("feature $(findfirst(v->isnan(v), features)) is nan")
    end

    features
end
