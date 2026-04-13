import numpy as np

def velocity_features(window):
    velocity_feats = []

    velocity = np.diff(window, axis=0)
    vel_mean = np.mean(velocity, axis=0)
    vel_std = np.std(velocity, axis=0)
    vel_max = np.max(np.abs(velocity), axis=0)

    velocity_feats.append(vel_mean)
    velocity_feats.append(vel_std)
    velocity_feats.append(vel_max)
    return np.concatenate([vel_mean, vel_std, vel_max])