""" tracker3d/loader.py
This file contains loading functions to generate or retrieve training data from
.csv files, to be used in neural network models.
@author: Daniel Zurawski
@organization: Fermilab
"""

import numpy as np
import pandas as pd
from tracker3d.utils import to_categorical

# Suppress FutureWarnings about np.full function.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# How to sort hits. Sort by "phi" first, then by "r" and finally by "z".
ORDERING  = ["phi", "r", "z"]

def dataload(frame, nev, tpe, ts, npe, z_bounds=(-200,200), verbose=False):
    """ Load input and output data from 'frame'.
    
    Arguments:
        frame (pd.DataFrame):
            A frame to sample events from.
            Headers must include: ['cluster_id', 'phi', 'r', 'z'].
            Each row contains the information for a single hit.
            'cluster_id' -- integer that designates the track of this hit.
            'phi' -- The phi angle (radians) describing hit position.
            'r' -- The radius of the layer that this hit occured at.
            'z' -- The z value describing this hit's z coordinate.
        nev (int):
            "Number of Events" -- The number of events to generate.
        tpe (int):
            "Tracks per Event" -- How many tracks belong to each event.
        ts (int):
            "Track Size" -- The number of hits that belong to each track.
        npe (int):
            "Noise per Event" -- The number of noisy hits (hits that belong to
            no track) to include in each event.
        verbose (bool):
            Whether or not this function should output notes on errors.
            True if you want messages. False if you don't want messages.

    Returns:
        A tuple: (train, target)
            train  is a numpy array of shape: (nev, ts*tpe+npe, 3)
            target is a numpy array of shape: (nev, ts*tpe+npe, tpe+1)
    """
    hpe    = (tpe * ts) + npe # Hits per event.
    train  = np.zeros((nev, hpe, len(ORDERING))) # Will be returned later.
    target = np.zeros((nev, hpe, tpe+1)) # Will be returned later
    layers = np.sort(np.partition(pd.unique(frame.r), ts-1)[:ts])
    hits   = frame[frame.r.isin(layers)]
    events = [event for (_, event)in hits.groupby("event_id")]
    
    wins = 0 # The number of successful event extractions.
    for i, event in enumerate(events):
        if wins >= nev:
            break
        try:
            goods = event.groupby("cluster_id").filter(lambda t: len(t) == ts)
            goods = goods.sort_values("cluster_id")[:ts * tpe]
            noise = _make_some_noise(npe, z_bounds, layers, tpe)
            lowlr = goods[goods.r == layers[0]].sort_values(ORDERING)
            ID2I  = dict((ID, i) for i, ID in enumerate(lowlr.cluster_id))
            goods.cluster_id = goods.cluster_id.map(ID2I)
            ehits        = pd.concat([goods, noise]).sort_values(ORDERING)
            train[wins]  = ehits[ORDERING].values
            target[wins] = to_categorical(ehits["cluster_id"].values.astype(np.uint), tpe + 1)            
            wins += 1
        except ValueError:
            if verbose:
                print("Failed reading event index {}.".format(i))
    if verbose:
        print("All finished. Loaded in {0} / {1}".format(wins, nev))
    return (train[:wins], target[:wins])
### END FUNCTION dataload

def var_dataload(frame, nev, tpe, ts, npe, z_bounds=(-200,200), verbose=False):
    """ Load input and output data from 'frame'.
    
    Arguments:
        frame (pd.DataFrame):
            A frame to sample events from.
            Headers must include: ['cluster_id', 'phi', 'r', 'z'].
            Each row contains the information for a single hit.
            'cluster_id' -- integer that designates the track of this hit.
            'phi' -- The phi angle (radians) describing hit position.
            'r' -- The radius of the layer that this hit occured at.
            'z' -- The z value describing this hit's z coordinate.
        nev (int):
            "Number of Events" -- The number of events to generate.
        tpe (int):
            "Tracks per Event" -- How many tracks belong to each event.
        ts (int):
            "Track Size" -- The number of hits that belong to each track.
        npe (int):
            "Noise per Event" -- The number of noisy hits (hits that belong to
            no track) to include in each event.
        verbose (bool):
            Whether or not this function should output notes on errors.
            True if you want messages. False if you don't want messages.

    Returns:
        A tuple: (train, target)
            train  is a numpy array of shape: (nev, ts*tpe+npe, 3)
            target is a numpy array of shape: (nev, ts*tpe+npe, tpe+1)
    """
    hpe    = (tpe * ts) + npe # Hits per event.
    train  = [] # Will be returned later.
    target = [] # Will be returned later
    layers = np.sort(np.partition(pd.unique(frame.r), ts-1)[:ts])
    hits   = frame[frame.r.isin(layers)]
    events = [event for (_, event)in hits.groupby("event_id")]

    wins = 0 # The number of successful event extractions.
    for i, event in enumerate(events):
        if wins >= nev:
            break
        try:
            goods = event.groupby("cluster_id").filter(lambda t: len(t) <= ts)
            IDs = np.unique(goods["cluster_id"].values)
            uniqueIDs = np.array([ID for ID in IDs if ID < tpe])
            goods = goods[goods.cluster_id.isin(uniqueIDs)]
            noise = _make_some_noise(npe, z_bounds, layers, tpe)
            lowlr = goods[goods.r == layers[0]].sort_values(ORDERING)
            ID2I  = dict((ID, i) for i, ID in enumerate(lowlr.cluster_id))
            goods.cluster_id = goods.cluster_id.map(ID2I)
            ehits        = pd.concat([goods, noise]).sort_values(ORDERING)
            train.append(ehits[ORDERING].values)
            target.append(to_categorical(ehits["cluster_id"].values.astype(np.uint), tpe + 1))
            wins += 1
        except ValueError:
            if verbose:
                print("Failed reading event index {}.".format(i))
    if verbose:
        print("All finished. Loaded in {0} / {1}".format(wins, nev))
    return (train[:wins], target[:wins])
### END FUNCTION dataload

def _make_some_noise(npe, z_bounds, layers, cluster_id):
    """ Make a pd.DataFrame of random noise hits. """
    noise = np.zeros((npe, 5)) # (event_id, cluster_id, phi, r, z)
    columns    = ["event_id", "cluster_id", "phi", "r", "z"]
    noise[:,0] = np.full(npe, -1) # event_id does not matter. Make it -1.
    noise[:,1] = np.full(npe, cluster_id)
    noise[:,2] = np.random.uniform(-np.pi, np.pi, npe)
    noise[:,3] = np.random.choice(layers, size=npe)
    noise[:,4] = np.random.uniform(z_bounds[0], z_bounds[1], npe)
    frame = pd.DataFrame(data=noise, columns=columns)
    return frame
### END FUNCTION _make_some_noise

if __name__ == "__main__":
    np.random.seed(7)
    frame = pd.read_csv("../datasets/standard_100MeV.csv")
    train, target = dataload(frame, nev=100, tpe=4, ts=4, npe=3)
    #print(train)
    #print(target)