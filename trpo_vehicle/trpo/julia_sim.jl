export restart, step_forward, get_state

function restart(roadway::Roadway, render::Bool)
    # Create scene
    scene = Scene()
    push!(scene,Vehicle(VehicleState(VecSE2(5.0 + rand()*10.0,randn()*0.5,0.0), roadway, 20.0), 
                    VehicleDef(1, AgentClass.CAR, 4.826, 1.81)))
    push!(scene,Vehicle(VehicleState(VecSE2(0.,randn()*0.5,0.0), roadway, 10.0 + rand()*20.0), 
                        VehicleDef(2, AgentClass.CAR, 4.826, 1.81)))
    
    # Specify models
    context = IntegratedContinuous(1/10,3) #1/framrate
    models = Dict{Int, DriverModel}()

    if render
    	model1 = LatLonSeparableDriver(context, 
	                                ProportionalLaneTracker(σ=0.0, kp=5.0, kd=0.5), 
	                                IntelligentDriverModel(σ=0.0, v_des=21.0))
	    model2 = LatLonSeparableDriver(context, 
	                                ProportionalLaneTracker(σ=0.0, kp=5.0, kd=0.5), 
	                                StaticLongitudinalDriver())
    else
	    model1 = LatLonSeparableDriver(context, 
	                                ProportionalLaneTracker(σ=0.01, kp=5.0, kd=0.5), 
	                                IntelligentDriverModel(σ=0.2, v_des=21.0))
	    model2 = LatLonSeparableDriver(context, 
	                                ProportionalLaneTracker(σ=0.01, kp=5.0, kd=0.5), 
	                                StaticLongitudinalDriver())
	end
    
    # Placeholder for actions
    return scene, model1, model2
end
function step_forward(scene::Scene, roadway::Roadway, model1::DriverModel, model2::DriverModel, action::Float64)
    
    observe!(model1, scene, roadway, 1)
    context = action_context(model1)
    scene[1].state = propagate(scene[1], rand(model1), context, roadway)

    model2.mlon.a = action
    observe!(model2, scene, roadway, 2)
    context = action_context(model2)
    scene[2].state = propagate(scene[2], rand(model2), context, roadway)
    
    return get_state(scene)
end
function get_state(scene::Scene)
	# Extract state values
	d = scene[1].state.posF.s - scene[2].state.posF.s - scene[1].def.length/2. - scene[2].def.length/2.
	r = scene[2].state.v - scene[1].state.v
    s = scene[1].state.v
    return d, r, s
end