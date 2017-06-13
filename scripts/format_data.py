import h5py
import numpy as np
import argparse

path = '/home/deepdrive/trpo_vehicle/2d_drive_data/'

parser = argparse.ArgumentParser()
parser.add_argument('--trajdatas', type=int, nargs='+', default=[])

parser.add_argument('--use_multifeat', type=bool, default=False)

#(core, temporal, well_behaved, neighbor, carlidar_range, carlidar_range_rate, roadlidar_range, i)
#parser.add_argument('--start_feature_indices',type=int,nargs='+',default= [1,8,14,17,45,65,85,125])
parser.add_argument('--extract_core', type=bool, default=True)
parser.add_argument('--extract_temporal', type=bool, default=False)
parser.add_argument('--extract_well_behaved', type=bool, default=True)
parser.add_argument('--extract_neighbor_features', type=bool, default=False)
parser.add_argument('--extract_carlidar', type=bool, default=True)
parser.add_argument('--extract_roadlidar', type=bool, default=False)
parser.add_argument('--extract_carlidar_rangerate', type=bool, default=True)

# parser.add_argument('--carlidar_nbeams',type=int,default=0)
# parser.add_argument('--roadlidar_nbeams',type=int,default=0)
# parser.add_argument('--roadlidar_nlanes',type=int,default=0)

args = parser.parse_args()

tt_split = '../2d_drive_data/NGSIM_train_test_split.h5'
filename1 = 'data_trajdata_i101_trajectories-0750-0805.jld'
filename2 = 'data_trajdata_i101_trajectories-0805-0820.jld'
filename3 = 'data_trajdata_i101_trajectories-0820-0835.jld'
filename4 = 'data_trajdata_i80_trajectories-0400-0415.jld'
filename5 = 'data_trajdata_i80_trajectories-0500-0515.jld'
filename6 = 'data_trajdata_i80_trajectories-0515-0530.jld'
all_filenames = [filename1, filename2,
                 filename3, filename4, filename5, filename6]

#trajdata_ix = 1
SEED = 456
MAX_TRAJ_LEN = 100
# TRAJS_PER_FILE=1000
np.random.seed(SEED)

# if trajdata_ix == 0:
#filenames= all_filenames
# else:
#filenames= [all_filenames[trajdata_ix - 1]]

#trajdatas= [1,2,3,4,5,6]
trajdatas = args.trajdatas
filenames = [all_filenames[t - 1] for t in trajdatas]

assign, indices, train_assign = None, None, None
if filenames == []:
    filenames = ['data_trajdata_passive_aggressive1.jld',
                 'data_trajdata_passive_aggressive2.jld']
else:
    with h5py.File(tt_split, 'r') as f:

        assign = f['data']['assignment'][...]
        indices = f['data']['trajdata_indeces'][...]
        train_assign = assign == 1

obs_B_T_Dos = []
act_B_T_Das = []
len_Bs = []

#feat_ix = map(lambda x : x - 1, [1,8,14,17,45,65,85,125])
feat_ix = map(lambda x: x, [0, 8, 14, 17, 45, 65, 85, 125])
# parser.add_argument('--extract_core',type=bool,default=False)
# parser.add_argument('--extract_temporal',type=bool,default=False)
# parser.add_argument('--extract_well_behaved',type=bool,default=False)
# parser.add_argument('--extract_neighbor_features',type=bool,default=False)
# parser.add_argument('--extract_carlidar',type=bool,default=False)
# parser.add_argument('--extract_roadlidar',type=bool,default=False)
# parser.add_argument('--extract_carlidar_rangerate',type=bool,default=False)

core_ixs = range(feat_ix[0], feat_ix[1])
temp_ixs = range(feat_ix[1], feat_ix[2])
well_ixs = range(feat_ix[2], feat_ix[3])
neig_ixs = range(feat_ix[3], feat_ix[4])
carl_ixs = range(feat_ix[4], feat_ix[5])
clrr_ixs = range(feat_ix[5], feat_ix[6])
roal_ixs = range(feat_ix[6], feat_ix[7])

get_ixs = []
if args.extract_core:
    get_ixs += core_ixs
if args.extract_temporal:
    get_ixs += temp_ixs
if args.extract_well_behaved:
    get_ixs += well_ixs
if args.extract_neighbor_features:
    get_ixs += neig_ixs
if args.extract_carlidar:
    get_ixs += carl_ixs
if args.extract_carlidar_rangerate:
    get_ixs += clrr_ixs
if args.extract_roadlidar:
    get_ixs += roal_ixs

#(core, temporal, well_behaved, neighbor, carlidar_range, carlidar_range_rate, roadlidar_range, i)

for i, filename in enumerate(filenames):
    julia_i = i + 1

    if args.use_multifeat:
        filename = "multifeat_" + filename

    with h5py.File(path + filename, 'r') as f:
        intervals = f['intervals'][...]
        targets = f['targets'][...]
        features = f['features'][...]
        if args.use_multifeat:
            features = features[:, get_ixs]

        # Compute trajectory lengths from interval values.
        shift_intervals = np.concatenate(
            (intervals[1:], np.array(features.shape[0])[None]))
        lens = shift_intervals - intervals

        trajs_obs = np.array(np.split(features, np.cumsum(lens[:]))[:-1])
        trajs_act = np.array(np.split(targets, np.cumsum(lens[:]))[:-1])

        # find training assignments in the current file.
        if indices is None:
            f_traj_train_assign = np.ones_like(lens)
        else:
            f_traj_train_assign = train_assign[indices == julia_i]
        f_tran_train_assign = np.repeat(f_traj_train_assign, lens)

        # extract trajectories and transitions based on assign data.
        ##train_features = features[f_tran_train_assign]
        ##train_targets = targets[f_tran_train_assign]
        train_lens = lens[f_traj_train_assign]
        train_trajs_obs = trajs_obs[f_traj_train_assign]
        train_trajs_act = trajs_act[f_traj_train_assign]

        #train_traj_ixs = traj_ixs[f_tran_train_assign]

        # sample one subtrajectory each.
        print("Training trajs in %s : %i" % (filename, len(train_lens)))
        start_ixs = np.array(map(np.random.randint, train_lens - MAX_TRAJ_LEN))
        s_train_trajs_obs = [traj[start_ix:start_ix + MAX_TRAJ_LEN][None, ...]
                             for start_ix, traj in zip(start_ixs, train_trajs_obs)]
        s_train_trajs_act = [traj[start_ix:start_ix + MAX_TRAJ_LEN][None, ...]
                             for start_ix, traj in zip(start_ixs, train_trajs_act)]

        o_B_T_Do = np.concatenate(s_train_trajs_obs, axis=0)
        a_B_T_Da = np.concatenate(s_train_trajs_act, axis=0)

        obs_B_T_Dos.append(o_B_T_Do)
        act_B_T_Das.append(a_B_T_Da)
        len_Bs.append(train_lens)

        #halt= True
        #train_intervals = intervals

        # with h5py.File('../expert_trajs/features%i_seed%i_mtl%i_ntraj%i_openaiformat.h5'%(Do,SEED,MAX_TRAJ_LEN,NUM_TRAJ), 'w') as hf:
        #hf.create_dataset('obs_B_T_Do', data= obs_B_T_Do)
        #hf.create_dataset('a_B_T_Da', data= a_B_T_Da)
        #hf.create_dataset('r_B_T', data= r_B_T)
        #hf.create_dataset('len_B', data= len_B)

obs_B_T_Do = np.concatenate(obs_B_T_Dos, axis=0)
act_B_T_Da = np.concatenate(act_B_T_Das, axis=0)
len_B = np.concatenate(len_Bs)

B, T, Do = obs_B_T_Do.shape

# if args.use_multifeat:
#name = '../expert_trajs/radar_features%i_mtl%i_seed%i_trajdata%s_openaiformat.h5'%(Do,T,SEED,''.join(map(str,trajdatas)))
# else:
#name = '../expert_trajs/features%i_mtl%i_seed%i_trajdata%s_openaiformat.h5'%(Do,T,SEED,''.join(map(str,trajdatas)))

# parser.add_argument('--extract_core',type=bool,default=False)
# parser.add_argument('--extract_temporal',type=bool,default=False)
# parser.add_argument('--extract_well_behaved',type=bool,default=False)
# parser.add_argument('--extract_neighbor_features',type=bool,default=False)
# parser.add_argument('--extract_carlidar',type=bool,default=False)
# parser.add_argument('--extract_roadlidar',type=bool,default=False)
# parser.add_argument('--extract_carlidar_rangerate',type=bool,default=False)

name = '../expert_trajs/core{}_temp{}_well{}_neig{}_carl{}_roal{}_clrr{}_mtl{}_seed{}.h5'.format(
    int(args.extract_core), int(args.extract_temporal), int(
        args.extract_well_behaved),
    int(args.extract_neighbor_features), int(
        args.extract_carlidar), int(args.extract_roadlidar),
    int(args.extract_carlidar_rangerate), MAX_TRAJ_LEN, SEED
)

with h5py.File(name, 'w') as hf:
    hf.create_dataset('obs_B_T_Do', data=obs_B_T_Do)
    hf.create_dataset('a_B_T_Da', data=act_B_T_Da)
    #hf.create_dataset('r_B_T', data= r_B_T)
    hf.create_dataset('len_B', data=len_B)

halt = True
