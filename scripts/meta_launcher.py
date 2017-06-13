import numpy as np


S = []
for policy_type in ["gru"]:
    for hspec, gru_dim in [  # ([128, 128, 128, 64, 64],64),
        #([256,128,64,64,32],32)]:
        #([256,256],64),
        #([256,256,32],32)]:
        ([256, 128, 64, 64, 32, 16], 16),
        ([512, 256, 128, 64, 32, 16], 16),
        ([512, 256, 128, 64, 64], 32),
        ([256, 128, 64, 64, 32, 32, 16, 16, 8, 8, 8], 8),
    ]:
        #1*256 + 1*128 + 2*64 + 2*32
        #2*256 + 1*64
        #256 + 32 +32
        for extract_temporal, temporal_noise_thresh in [(0, 0)]:
            for max_traj_len in [100]:
                exp_name = "gru_temp{}_mtl{}".format(
                    extract_temporal, max_traj_len)
                s = 'python scripts/train_gail_model.py --nonlinearity "elu" --gail_batch_size 1024 --reward_type "mlp" ' +\
                    '--feature_type "mlp" --policy_type {policy_type} --baseline_type "linear" --gru_dim {gru_dim} --normalize_obs 1 ' +\
                    '--decay_rate 1.0 --decay_steps 1 --n_iter 500 --adam_lr 0.00005 --hspec{hspec}' +\
                    '--env_name "Auto2D" --extract_core 1 --extract_well_behaved 1 --extract_temporal {extract_temporal} ' + \
                    '--extract_neighbor_features 0 --extract_carlidar 1 --extract_carlidar_rangerate 1 --exp_name {exp_name} ' +\
                    '--max_traj_len {max_traj_len} --temporal_noise_thresh {temporal_noise_thresh}; '

                s = s.format(policy_type=policy_type,
                             gru_dim=gru_dim,
                             hspec=hspec,
                             extract_temporal=extract_temporal, temporal_noise_thresh=temporal_noise_thresh,
                             max_traj_len=max_traj_len,
                             exp_name=exp_name)
                s.replace('[', ' ')
                s.replace(']', ' ')

                S.append(s)

S = ''.join(S)
S = S.replace('[', ' ')
S = S.replace(']', ' ')
S = S.replace(',', '')

print(S)
