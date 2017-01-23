using AutomotiveDrivingModels
using AutoViz

#############################################################
# Lidar Sensor

type LidarSensor
    angles::Vector{Float64}
    ranges::Vector{Float64}
    range_rates::Vector{Float64}
    max_range::Float64
    poly::ConvexPolygon
end
function LidarSensor(nbeams::Int;
    max_range::Float64=100.0,
    angle_offset::Float64=0.0,
    )


    if nbeams > 1
        angles = collect(linspace(angle_offset,2*pi+angle_offset,nbeams+1))[2:end]
    else
        angles = Float64[]
        nbeams = 0
    end

    ranges = Array(Float64, nbeams)
    range_rates = Array(Float64, nbeams)
    LidarSensor(angles, ranges, range_rates, max_range, ConvexPolygon(4))
end
nbeams(lidar::LidarSensor) = length(lidar.angles)
function observe!(lidar::LidarSensor, scene::Scene, roadway::Roadway, vehicle_index::Int)
    state_ego = scene[vehicle_index].state
    egoid = scene[vehicle_index].def.id
    ego_vel = polar(state_ego.v, state_ego.posG.θ)
    for (i,angle) in enumerate(lidar.angles)
        ray_angle = state_ego.posG.θ + angle
        ray_vec = polar(1.0, ray_angle)
        ray = VecSE2(state_ego.posG.x, state_ego.posG.y, ray_angle)

        range = lidar.max_range
        range_rate = 0.0
        for veh in scene
            if veh.def.id != egoid
                to_oriented_bounding_box!(lidar.poly, veh)

                range2 = AutomotiveDrivingModels.get_collision_time(ray, lidar.poly, 1.0)
                if !isnan(range2) && range2 < range
                    range = range2
                    relative_speed = polar(veh.state.v, veh.state.posG.θ) - ego_vel
                    range_rate = proj(relative_speed, ray_vec, Float64)
                end
            end
        end
        lidar.ranges[i] = range
        lidar.range_rates[i] = range_rate
    end

    lidar
end
function render_lidar!(rendermodel::RenderModel, lidar::LidarSensor, posG::VecSE2;
    color::Colorant=colorant"white",
    line_width::Float64 = 0.05,
    )

    for (angle, range, range_rate) in zip(lidar.angles, lidar.ranges, lidar.range_rates)
        ray = VecSE2(posG.x, posG.y, posG.θ + angle)
        t = 2 / (1 + exp(-range_rate))
        ray_color = t > 1.0 ? lerp(RGBA(1.0,1.0,1.0,0.5), RGBA(0.2,0.2,1.0,0.5), t/2) :
                              lerp(RGBA(1.0,0.2,0.2,0.5), RGBA(1.0,1.0,1.0,0.5), t)
        render_ray!(rendermodel::RenderModel, ray, color=ray_color, length=range, line_width=line_width)
    end
    rendermodel
end

#############################################################
# Road Line Lidar Sensor

type RoadlineLidarSensor
    angles::Vector{Float64}
    ranges::Matrix{Float64} # [n_lanes by nbeams]
    max_range::Float64
    poly::ConvexPolygon
end
nbeams(lidar::RoadlineLidarSensor) = length(lidar.angles)
nlanes(lidar::RoadlineLidarSensor) = size(lidar.ranges, 1)
function RoadlineLidarSensor(nbeams::Int;
    max_range::Float64=100.0,
    max_depth::Int=2,
    angle_offset::Float64=0.0,
    )

    if nbeams > 1
        angles = collect(linspace(angle_offset,2*pi+angle_offset,nbeams+1))[2:end]
    else
        angles = Float64[]
        nbeams = 0
    end
    ranges = Array(Float64, max_depth, nbeams)
    RoadlineLidarSensor(angles, ranges, max_range, ConvexPolygon(4))
end
function _update_lidar!(lidar::RoadlineLidarSensor, ray::VecSE2, beam_index::Int, p_lo::VecE2, p_hi::VecE2)
    test_range = get_collision_time(ray, AutomotiveDrivingModels.LineSegment(p_lo, p_hi), 1.0)

    if !isnan(test_range)
        n_ranges = size(lidar.ranges, 1)
        for k in 1 : n_ranges
            if test_range < lidar.ranges[k,beam_index]
                for l in k+1 : n_ranges
                    lidar.ranges[l,beam_index] = lidar.ranges[l-1,beam_index]
                end
                lidar.ranges[k,beam_index] = test_range
                break
            end
        end
    end

    lidar
end
function _update_lidar!(lidar::RoadlineLidarSensor, ray::VecSE2, beam_index::Int, lane::Lane, roadway::Roadway; check_right_lane::Bool=false)
    halfwidth = lane.width/2
    Δ = check_right_lane ? -π/2 : π/2
    p_lo = convert(VecE2, lane.curve[1].pos + polar(halfwidth, lane.curve[1].pos.θ + Δ))
    for j in 2:length(lane.curve)
        p_hi = convert(VecE2, lane.curve[j].pos + polar(halfwidth, lane.curve[j].pos.θ + Δ))
        _update_lidar!(lidar, ray, beam_index, p_lo, p_hi)
        p_lo = p_hi
    end
    if has_next(lane)
        lane2 = next_lane(lane, roadway)
        pt = lane2.curve[1]
        p_hi = convert(VecE2, pt.pos + polar(lane2.width/2, pt.pos.θ + Δ))
        _update_lidar!(lidar, ray, beam_index, p_lo, p_hi)
    end
    lidar
end
function _update_lidar!(lidar::RoadlineLidarSensor, ray::VecSE2, beam_index::Int, roadway::Roadway)

    for seg in roadway.segments
        for lane in seg.lanes

            # always check the left lane marking
            _update_lidar!(lidar, ray, beam_index, lane, roadway)

            # only check the right lane marking if this is the first lane
            if lane.tag.lane == 1
                _update_lidar!(lidar, ray, beam_index, lane, roadway, check_right_lane=true)
            end
        end
    end

    lidar
end
function observe!(lidar::RoadlineLidarSensor, scene::Scene, roadway::Roadway, vehicle_index::Int)
    state_ego = scene[vehicle_index].state
    egoid = scene[vehicle_index].def.id
    ego_vel = polar(state_ego.v, state_ego.posG.θ)

    fill!(lidar.ranges, lidar.max_range)
    for (beam_index,angle) in enumerate(lidar.angles)
        ray_angle = state_ego.posG.θ + angle
        ray = VecSE2(state_ego.posG.x, state_ego.posG.y, ray_angle)

        _update_lidar!(lidar, ray, beam_index, roadway)
    end

    lidar
end
function render_lidar!(rendermodel::RenderModel, lidar::RoadlineLidarSensor, posG::VecSE2;
    color::Colorant=RGBA(1.0,1.0,1.0,0.5),
    line_width::Float64 = 0.05,
    depth_level::Int = 1, # if 1, render first lanes struck by beams, if 2 render 2nd ...
    )

    for (angle, range) in zip(lidar.angles, lidar.ranges[depth_level,:])
        ray = VecSE2(posG.x, posG.y, posG.θ + angle)
        render_ray!(rendermodel::RenderModel, ray, color=color, length=range, line_width=line_width)
    end
    rendermodel
end

#############################################################
# Road Line Lidar Culling

immutable LanePortion
    tag::LaneTag
    curveindex_lo::Int
    curveindex_hi::Int
end
type RoadwayLidarCulling
    is_leaf::Bool

    x_lo::Float64
    x_hi::Float64
    y_lo::Float64
    y_hi::Float64

    top_left::RoadwayLidarCulling
    top_right::RoadwayLidarCulling
    bot_left::RoadwayLidarCulling
    bot_right::RoadwayLidarCulling

    lane_portions::Vector{LanePortion}

    function RoadwayLidarCulling(x_lo::Float64, x_hi::Float64, y_lo::Float64, y_hi::Float64, lane_portions::Vector{LanePortion})
        retval = new()
        retval.is_leaf = true
        retval.x_lo = x_lo
        retval.x_hi = x_hi
        retval.y_lo = y_lo
        retval.y_hi = y_hi
        retval.lane_portions = lane_portions
        retval
    end
    function RoadwayLidarCulling(x_lo::Float64, x_hi::Float64, y_lo::Float64, y_hi::Float64,
                                 top_left::RoadwayLidarCulling,
                                 top_right::RoadwayLidarCulling,
                                 bot_left::RoadwayLidarCulling,
                                 bot_right::RoadwayLidarCulling,)
        retval = new()
        retval.is_leaf = false
        retval.x_lo = x_lo
        retval.x_hi = x_hi
        retval.y_lo = y_lo
        retval.y_hi = y_hi
        retval.top_left = top_left
        retval.top_right = top_right
        retval.bot_left = bot_left
        retval.bot_right = bot_right
        retval
    end
end
RoadwayLidarCulling() = RoadwayLidarCulling(typemax(Float64), typemin(Float64), typemax(Float64), typemin(Float64), LanePortion[])

function Base.get(rlc::RoadwayLidarCulling, x::Real, y::Real)
    if rlc.is_leaf
        return rlc
    else
        if y < (rlc.y_lo + rlc.y_hi)/2
            if x < (rlc.x_lo + rlc.x_hi)/2
                return get(rlc.bot_left, x, y)
            else
                return get(rlc.bot_right, x, y)
            end
        else
            if x < (rlc.x_lo + rlc.x_hi)/2
                return get(rlc.top_left, x, y)
            else
                return get(rlc.top_right, x, y)
            end
        end
    end
end
function ensure_leaf_in_rlc!(rlc::RoadwayLidarCulling, x::Real, y::Real, fidelity_x::Real, fidelity_y::Real)

    leaf = get(rlc, x, y)
    @assert(leaf.is_leaf)

    x_lo, x_hi = leaf.x_lo, leaf.x_hi
    y_lo, y_hi = leaf.y_lo, leaf.y_hi
    while (x_hi - x_lo) > fidelity_x || (y_hi - y_lo) > fidelity_y # drill down

        x_mid = (x_hi + x_lo)/2
        y_mid = (y_hi + y_lo)/2

        leaf.top_left  = RoadwayLidarCulling(x_lo, x_mid, y_mid, y_hi, LanePortion[])
        leaf.top_right = RoadwayLidarCulling(x_mid, x_hi, y_mid, y_hi, LanePortion[])
        leaf.bot_left  = RoadwayLidarCulling(x_lo, x_mid, y_lo, y_mid, LanePortion[])
        leaf.bot_right = RoadwayLidarCulling(x_mid, x_hi, y_lo, y_mid, LanePortion[])
        leaf.is_leaf = false

        leaf = get(leaf, x, y)
        x_lo, x_hi = leaf.x_lo, leaf.x_hi
        y_lo, y_hi = leaf.y_lo, leaf.y_hi
    end

    rlc
end
function get_lane_portions(roadway::Roadway, x::Real, y::Real, lane_portion_max_range::Float64)
    P = VecE2(x, y)
    Δ² = lane_portion_max_range*lane_portion_max_range

    lane_portions = LanePortion[]

    for seg in roadway.segments
        for lane in seg.lanes
            f = curvept -> abs2(curvept.pos - P) ≤ Δ²
            i = findfirst(f, lane.curve)
            if i != 0
                j = findlast(f, lane.curve)
                @assert(j != 0)
                push!(lane_portions, LanePortion(lane.tag, i, j))
            end
        end
    end

    lane_portions
end

function RoadwayLidarCulling(
    roadway::Roadway,
    lane_portion_max_range::Float64, # lane portions will be extracted such that points within lane_portion_max_range are in the leaves
    culling_fidelity::Float64, # the culling will drill down until Δx, Δy < culling_fidelity
    )

    #=
    1 - get area bounds
    2 - ensure all points in rlc
    3 - for each leaf, construct all lane portions
    =#

    # get area bounds
    x_lo = typemax(Float64)
    x_hi = typemin(Float64)
    y_lo = typemax(Float64)
    y_hi = typemin(Float64)
    for seg in roadway.segments
        for lane in seg.lanes
            for curvept in lane.curve
                pos = curvept.pos
                x_lo = min(pos.x, x_lo)
                x_hi = max(pos.x, x_hi)
                y_lo = min(pos.y, y_lo)
                y_hi = max(pos.y, y_hi)
            end
        end
    end
    x_lo -= lane_portion_max_range
    x_hi += lane_portion_max_range
    y_lo -= lane_portion_max_range
    y_hi += lane_portion_max_range
    root = RoadwayLidarCulling(x_lo, x_hi, y_lo, y_hi, LanePortion[])

    # ensure all points are in rlc
    x = x_lo - culling_fidelity
    while x < x_hi
        x += culling_fidelity
        y = y_lo - culling_fidelity
        while y < y_hi
            y += culling_fidelity
            ensure_leaf_in_rlc!(root, x, y, culling_fidelity, culling_fidelity)
        end
    end

    nodes = RoadwayLidarCulling[root]
    while !isempty(nodes)
        node = pop!(nodes)
        if node.is_leaf
            x = (node.x_lo + node.x_hi)/2
            y = (node.y_lo + node.y_hi)/2
            append!(node.lane_portions, get_lane_portions(roadway, x, y, lane_portion_max_range))
        else
            push!(nodes, node.top_left)
            push!(nodes, node.top_right)
            push!(nodes, node.bot_left)
            push!(nodes, node.bot_right)
        end
    end

    root
end

function _update_lidar!(
    lidar::RoadlineLidarSensor,
    ray::VecSE2,
    beam_index::Int,
    roadway::Roadway,
    lane_portion::LanePortion;
    check_right_lane::Bool=false
    )

    lane = roadway[lane_portion.tag]

    halfwidth = lane.width/2
    Δ = check_right_lane ? -π/2 : π/2
    p_lo = convert(VecE2, lane.curve[lane_portion.curveindex_lo].pos +
                          polar(halfwidth, lane.curve[lane_portion.curveindex_lo].pos.θ + Δ))
    for j in lane_portion.curveindex_lo+1:lane_portion.curveindex_hi
        p_hi = convert(VecE2, lane.curve[j].pos + polar(halfwidth, lane.curve[j].pos.θ + Δ))
        _update_lidar!(lidar, ray, beam_index, p_lo, p_hi)
        p_lo = p_hi
    end
    if lane_portion.curveindex_hi == length(lane.curve) && has_next(lane)
        lane2 = next_lane(lane, roadway)
        pt = lane2.curve[1]
        p_hi = convert(VecE2, pt.pos + polar(lane2.width/2, pt.pos.θ + Δ))
        _update_lidar!(lidar, ray, beam_index, p_lo, p_hi)
    end
    lidar
end
function _update_lidar!(lidar::RoadlineLidarSensor, ray::VecSE2, beam_index::Int, roadway::Roadway, rlc::RoadwayLidarCulling)

    leaf = get(rlc, ray.x, ray.y)
    for lane_portion in leaf.lane_portions

        # always update the left lane marking
        _update_lidar!(lidar, ray, beam_index, roadway, lane_portion)

        # only check the right lane marking if this is the first lane
        if lane_portion.tag.lane == 1
            _update_lidar!(lidar, ray, beam_index, roadway, lane_portion, check_right_lane=true)
        end
    end
    lidar
end
function observe!(lidar::RoadlineLidarSensor, scene::Scene, roadway::Roadway, vehicle_index::Int, rlc::RoadwayLidarCulling)
    state_ego = scene[vehicle_index].state
    egoid = scene[vehicle_index].def.id
    ego_vel = polar(state_ego.v, state_ego.posG.θ)

    fill!(lidar.ranges, lidar.max_range)
    for (beam_index,angle) in enumerate(lidar.angles)
        ray_angle = state_ego.posG.θ + angle
        ray = VecSE2(state_ego.posG.x, state_ego.posG.y, ray_angle)

        _update_lidar!(lidar, ray, beam_index, roadway, rlc)
    end

    lidar
end



