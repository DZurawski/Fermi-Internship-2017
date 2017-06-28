""" tracker3d/loader.py
This file contains loading functions to generate or retrieve training data from
.csv files, to be used in neural network models.
@author: Daniel Zurawski
@organization: Fermilab
"""

import numpy as np
import pandas as pd
from keras.utils import to_categorical

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ORDERING  = ["phi", "r", "z"]

def dataload(frame, nev, tpe, ts, npe):
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

    Returns:
        A tuple: (train, target)
            train  is a numpy array of shape: (nev, ts*tpe+npe, 3)
            target is a numpy array of shape: (nev, ts*tpe+npe, tpe+1)
    """
    hpe    = (tpe * ts) + npe # Hits per event.
    train  = np.zeros((nev, hpe, len(ORDERING)))
    target = np.zeros((nev, hpe, tpe+1))
    layers = np.sort(np.partition(pd.unique(frame.r), ts-1)[:ts])
    hits   = frame[frame.r.isin(layers)]
    events = [event for (_, event)in hits.groupby("event_id")][:nev]
    
    for i, event in enumerate(events):
        try:
            goods = event.groupby("cluster_id").filter(lambda t: len(t) == ts)
            goods = goods.sort_values("cluster_id")[:ts * tpe]
            noise = _make_some_noise(npe, (-200, 200), layers, tpe)
            lowlr = goods[goods.r == layers[0]].sort_values(ORDERING)
            ID2I  = dict((ID, i) for i, ID in enumerate(lowlr.cluster_id))
            goods.cluster_id = goods.cluster_id.map(ID2I)
            ehits     = pd.concat([goods, noise]).sort_values(ORDERING)
            train[i]  = ehits[ORDERING].values
            target[i] = to_categorical(ehits["cluster_id"].values, tpe + 1)
        except ValueError as ve:
            print("Bad mojo at event index {}!".format(i))
            print(ve)
            print("Dan's Note: This is probably occurring because there")
            print("are not enough clusters with enough hits in them. Some")
            print("clusters do not have a layer 0 hit, and so they will not")
            print("have enough hits.")
    return (train, target)
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