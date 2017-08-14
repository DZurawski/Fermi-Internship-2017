""" tracker3d/loader.py

This file contains loading functions to generate or retrieve training data
from .csv files, to be used in neural network models.

Author: Daniel Zurawski
Author: Keshav Kapoor
Organization: Fermilab
Grammar: Python 3.6.1
"""

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import csr_matrix
from typing import Tuple, Sequence, Optional
from .tracker_types import Train, Target, Event, PMatrix
from . import utils


def from_frame(frame: pd.DataFrame,
               nev: int=-1,  # "Max Number of Events"
               tpe: int=-1,  # "Max Tracks Per Event"
               ts: int=-1,   # "Max Track Size"
               order: Tuple[str, str, str]=("phi", "z", "r"),
               n_noise: int=0,
               preferred_rows: Optional[int]=None,
               preferred_tracks: Optional[int]=None,
               event_range: Optional[Tuple[int, int]]=None)\
        -> Tuple[Train, Target]:
    """ Load input and output data from *frame*.

    Usually, some of these hits will be padding: (0, 0, 0) entries.
    The last column in the Target's PMatrix's correspond to the
    padding category. If *n_noise* is greater than 0, then each Event in the
    Train input data will have *n_noise* number of noisy hits. Noisy hits
    belong to no track. The second to last column in the PMatrix's correspond
    to the noise category.

    Arguments:
        frame (pd.DataFrame):
            A frame to sample events from.
            Headers must include:
            ['event_id', 'cluster_id', 'phi', 'r', 'z'].
            Each row contains the information for a single hit.
            'event_id' -- integer that designates a hit to an event.
            'cluster_id' -- integer that designates the track of this hit.
            'phi' -- The phi angle (radians) describing hit position.
            'r' -- The radius of the layer that this hit occurred at.
            'z' -- The z value describing this hit's z coordinate.
        nev (int):
            "Number of Events" -- The max number of events to generate.
            Will generate up to *nev* events, but it is possible that
            less events are generated, if the *frame* does not contain
            enough valid events. If nev < 0, then load in all events.
        tpe (int):
            "Tracks per Event" -- Max num of tracks that belong to each event.
            If tpe < 0, then there will not be a max.
        ts (int):
            "Track Size" -- Max number of hits that belong to each track.
            If ts < 0, then there will not be a max.
        order (Tuple[str, str, str]):
            The order in which hits should be arranged. It should be some
            permutation of ("phi", "r", "z").
        n_noise (int):
            How much noise to add per event. This should be non-negative.
        preferred_rows (Optional[int]):
            The number of rows in all events. Padding with zero entries will be
            used to ensure all events have the same number of rows.
        preferred_tracks (Optional[int]):
            The number of tracks per event. If not specified, defaults to max
            tracks per event among all events. If you decide to specify this,
            remember to add enough columns to accommodate the noise category
            and the padding category!
        event_range:
            The range used to take a subset from the events.

    Returns: (Tuple[Train, Target])
        Train  : np.array with 3 dimensions.
            shape = (number of loaded-in events, hits per event, 3)
        Target : np.array with 3 dimensions.
            shape = (number of loaded-in events, hits per event, categories)
        Each Event in Train will be padded with rows of 0's at the end so that
        each Event is the same shape. Also, each PMatrix in Target will consist
        of (n + 2) columns, where n is the max number of tracks in any event.
    """
    ts     = len(pd.unique(frame["r"])) if (ts < 0) else ts
    order  = list(order)  # Will absolutely need *order* to be a list for pd.
    train  = []  # Will contain hit position arrays.
    target = []  # Will contain target probability matrices.
    hits   = frame[frame.r.isin(_get_lowest_uniques(frame.r, ts))]
    events = [event for (_, event) in hits.groupby(by="event_id", sort=False)]
    max_tk = max([len(e.groupby("cluster_id")) for e in events])
    nev    = len(events) if (nev < 0) else nev
    tpe    = max_tk if (tpe < 0) else min([tpe, max_tk])
    n_cat  = tpe + 2 if preferred_tracks is None else preferred_tracks
    if event_range is None:
        space = range(min([len(events), nev]))
    else:
        space = range(min([event_range[0], len(events)]),
                      min([event_range[1], len(events)]))
    for i in space:
        # Get eligible tracks.
        ids   = _get_lowest_uniques(events[i].cluster_id, tpe)
        goods = events[i][events[i].cluster_id.isin(ids)]
        noise = _make_noise(n_noise, goods, n_cat)

        # Adjust each cluster_id to an index within a probability matrix.
        idx  = goods.groupby(["cluster_id"])["r"].transform(min) == goods["r"]
        lows = goods[idx].sort_values(["phi", "z", "r"])
        id2i = dict((_id, i) for i, _id in enumerate(lows.cluster_id))
        goods.cluster_id = goods.cluster_id.map(id2i)

        # Finally, append this event to the list of trains and targets.
        concat = pd.concat([goods, noise])
        sortie = concat.sort_values(order)
        tracks = sortie.cluster_id
        train.append(sortie[order].values)
        target.append(utils.to_categorical(tracks.values, n_cat))
    if event_range is None:
        print("All finished. Loaded in {}.".format(min([len(events), nev])))
    else:
        print("All finished. Loaded in range [{}, {}].".format(*event_range))
    return _padded_train_and_target(train, target, preferred_rows)


def _get_eligible_events(frame: pd.DataFrame, ts: int=-1):
    ts = len(pd.unique(frame.r)) if (ts < 0) else ts
    hits = frame[frame.r.isin(_get_lowest_uniques(frame.r, ts))]
    return [event for (_, event) in hits.groupby("event_id")]


def to_file(train: Train,
            target: Target,
            filename: str,
            sparse: bool=True)\
        -> None:
    """ Save training and target information to a .npz file.

    Arguments:
        train (Train):
            The training data to save.
        target (Target):
            The target probability matrices to save.
        filename (str):
            The name of the file to save to.
        sparse (bool):
            True if target matrices should be saved in csr format. This format
            saves a great deal of space for matrices with many zero entries.
            False if you just want to save the target as a regular matrix.

    Returns: (None)
    """
    if sparse:
        csr = [csr_matrix(matrix) for matrix in target]
        np.savez(filename,
                 train=train,
                 data=np.array([matrix.data for matrix in csr]),
                 indices=np.array([matrix.indices for matrix in csr]),
                 indptr=np.array([matrix.indptr for matrix in csr]),
                 shape=np.array([matrix.shape for matrix in csr]))
    else:
        np.savez(filename, train=train, target=target)


def from_file(filename: str,
              sparse: bool=True)\
        -> Tuple[Train, Target]:
    """ Retrieve training data and target data from a .npz file.

    Arguments:
        filename (str):
            The name of the file to extract from. It must be a .npz file.
        sparse (bool):
            True if the file's target data are in csr sparse matrix
            representation. Else, False.

    Returns: (Tuple[Train, Target])
        A Tuple consisting of the extracted training data and target data.
    """
    loader = np.load(filename)
    if sparse:
        train   = loader["train"]
        data    = loader["data"]
        indices = loader["indices"]
        indptr  = loader["indptr"]
        shape   = loader["shape"]
        target  = np.array([csr_matrix((data[i], indices[i], indptr[i]),
                                       shape=shape[i]).todense()
                            for i in range(len(data))])
        return train, target
    else:
        return loader["train"], loader["target"]


def _get_lowest_uniques(frame: pd.DataFrame,
                        max_size: int)\
        -> np.ndarray:
    """ The the lowest *max_size* number of elements from the frame.

    Arguments:
        frame (pd.DataFrame):
            A DataFrame object with 1 column.
        max_size (int):
            The maximum number of objects to return. If the *frame* contains
            less objects than max_size, then return the entire frame.
    Returns: (np.ndarray)
        A sorted array of integer values.
    """
    uniques  = pd.unique(frame)
    n_unique = np.min([len(uniques), max_size])
    return np.sort(np.partition(uniques, n_unique - 1))[:n_unique]


def _padded_train_and_target(train: Sequence[Event],
                             target: Sequence[PMatrix],
                             preferred_size: int)\
        -> Tuple[Train, Target]:
    """ Pad *train* and *target*. Return the padded Train and Target.

    Pad *train* and *target* such that each Event in *train* has the same
    number of rows and eac PMatrix in *target* has the same number of rows.
    The padded rows in the returned PMatrix's will have 1's in the final
    column, where this column represents the padding category.

    Arguments:
        train (Train):
            training data to be padded.
        target (Target):
            target data to be padded.

    Return: (Tuple[Train, Target])
        The padded Train and Target data.
    """
    p_train  = pad_sequences(
            train,
            padding="post",
            maxlen=preferred_size,
            value=0,
            dtype=np.float32)
    p_target = pad_sequences(
            target,
            padding="post",
            maxlen=preferred_size,
            value=0,
            dtype=np.int32)

    # The final column in target is the "padding" category. For each padding
    # that we added, assign the column in its row to 1.
    for matrix in p_target:
        for row in matrix:
            if np.sum(row) == 0:
                row[-1] = 1

    return p_train, p_target


def _make_noise(n_noise: int,
                goods: pd.DataFrame,
                n_cat: int)\
        -> pd.DataFrame:
    """ Create a pd.DataFrame containing noisy hits.

    Noise is defined as a hit belonging to no track. Noise is generated by
    generating 3 aspects: *r*, *phi*, *z*.

    Generating *r*:
    For each noisy hit to be generated for the *goods* frame, create a list of
    all radiuses within *goods* and then uniform randomly choose one of these
    radiuses to be this particular noisy hit's radius.

    Generating *phi*:
    Uniform randomly choose a float between -pi and pi.

    Generating *z*:
    For *goods*, retrieve the minimum and maximum *z* values achieved by all
    of that event's hits. Uniform randomly choose a float between the
    minimum *z* and maximum *z*.

    Arguments:
        n_noise (int):
            Number of noise hits to generate.
        goods (pd.DataFrame):
            The frame of data to generate the data for.
        n_cat (int):
            The number of categories for this frame.

    Returns: (pd.DataFrame)
    """
    if n_noise < 1:
        return pd.DataFrame()
    rnge = range(n_noise)
    zbd  = (goods["z"].min(), goods["z"].max())  # Z bounds.
    return pd.DataFrame(data={
        "event_id": [-1 for _ in rnge],
        "cluster_id": tuple(n_cat - 2 for _ in rnge),
        "r": tuple(goods["r"].sample(n_noise, replace=True)),
        "phi": tuple(2 * np.pi * np.random.random() - np.pi for _ in rnge),
        "z": tuple((zbd[1]-zbd[0]) * np.random.random() + zbd[0] for _ in rnge)
    })
