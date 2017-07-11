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
from typing import Tuple, Sequence, List
from .tracker_types import Train, Target
from .utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import csr_matrix


def from_frame(frame: pd.DataFrame,
               nev: int,  # "Number of Events"
               tpe: int,  # "Tracks Per Event"
               ts: int,  # "Track Size"
               variable_data: bool=False,
               order: Sequence[str]= ("phi", "r", "z"),
               verbose: bool=True)\
        -> Tuple[Train, Target]:
    """ Load input and output data from *frame*.

    Each event returned has the same number of tracks and each track has the
    same number of corresponding hits. If variable_data is True, then some of
    these hits will be padding: (0, 0, 0) entries.

    Arguments:
        frame (pd.DataFrame):
            A frame to sample events from.
            Headers must include:
            ['event_id', 'cluster_id', 'phi', 'r', 'z'].
            Each row contains the information for a single hit.
            'event_id' -- integer that designates a hit to an event.
            'cluster_id' -- integer that designates the track of this hit.
            'phi' -- The phi angle (radians) describing hit position.
            'r' -- The radius of the layer that this hit occured at.
            'z' -- The z value describing this hit's z coordinate.
        nev (int):
            "Number of Events" -- The number of events to generate.
            Will generate up to *nev* events, but it is possible that
            less events are generated, if the *frame* does not contain
            enough valid events.
        tpe (int):
            "Tracks per Event" -- How many tracks belong to each event.
            If *variable_data* is True, then *tpe* is the upper limit for
            how many tracks belong to each event. Otherwise, all events
            have the same number of tracks assigned to them.
        ts (int):
            "Track Size" -- The number of hits that belong to each track.
            If *variable_data* is True, then *ts* is the upper limit for
            how many hits belong to each track. Otherwise, all tracks have
            the same number of hits assigned to them.
        variable_data (bool):
            True if we want to load in events with less tracks than *tpe*
                    and tracks with less hits than *ts*.
            False if we want all events to have the same number of tracks
                    and all tracks to have the same number of hits.
        order (Tuple[str]):
            The order in which hits should be arranged.
        verbose (bool):
            If True, print failures to standard out. Else, do not.

    Returns: (Tuple[Train, Target])
        Train  : np.array, shape(nev, (ts * tpe + npe), len(order))
        Target : np.array, shape(nev, (ts * tpe + npe),        tpe)
        If *variable_data* is True, then each Event in Train will be padded
        with rows of 0's at the end so that each Event is the same shape.
        Also, each PMatrix in Target will consist of (n + 1) columns,
        where n is the max number of tracks in any event. The final column
        is used as the 'padding' category, which is used to designate a
        padding row as padding.
    """
    order  = list(order)  # Will absolutely need *order* to be a list for pd.
    train  = []  # Will contain hit position arrays.
    target = []  # Will contain target probability matrices.

    # *layers* is an array of unique valid r values that hits can have.
    # Valid r values are the lowest *ts* number of r values.
    # If *ts* > (number of layers), then *layers* is all unique layers.
    layers = _get_lowest_uniques(frame.r, ts)

    # *hits* is a pd.DataFrame where rows contain hit info for a some hit.
    hits = frame[frame.r.isin(layers)]

    # Just a list of Events.
    events = [event for (_, event) in hits.groupby("event_id")]

    # The number of successful event extractions.
    wins = 0

    # Adjust *tpe* for *tpe* that are too large.
    if variable_data:
        tpe = min(tpe, max(frame.groupby(["event_id", "r"]).size()))

    # Function to be used to filter out the tracks with unwanted track size.
    def good_len(track: pd.DataFrame) -> bool:
        return (track.r.min() == layers[0]
                and (((len(track) <= ts) and variable_data)
                     or ((len(track) == ts) and not variable_data)))

    # Time to populate the *train* and *target* lists.
    for i, event in enumerate(events):
        if nev <= wins:
            break  # We have enough successful event extractions. Time to go.
        try:
            # Get eligible tracks.
            goods = event.groupby("cluster_id").filter(good_len)
            ids   = _get_lowest_uniques(goods.cluster_id, tpe)
            goods = goods[goods.cluster_id.isin(ids)]

            # Adjust each cluster_id to an index within a probability matrix.
            _assign_cluster_id_to_matrix_index(goods, order, layers[0])

            # Finally, append this event to the list of trains and targets.
            sortie = goods.sort_values(order)
            tracks = sortie.cluster_id
            train.append(sortie[order].values)
            target.append(to_categorical(tracks.values, tpe + 1)) # +1 for pad
            wins += 1
        except ValueError:
            if verbose:
                print("Failed reading event index {}.".format(i))
    if verbose:
        print("All finished. Loaded in {0} / {1}".format(wins, nev))

    # Pad the sequences to allow for variable number of hits per event
    # and variable number of tracks per event.
    return _padded_train_and_target(train[:wins], target[:wins])


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
            True if target matrices should be saved in csr format.
            This format saves a great deal of space for matrices with many
            zero entries.
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


def _assign_cluster_id_to_matrix_index(frame: pd.DataFrame,
                                       order: List[str],
                                       min_r: int)\
        -> None:
    """ Assign the cluster_id within the frame to a matrix index.

    Arguments:
        frame (pd.DataFrame):
            A data frame with a cluster_id column.
        order (List[str]):
            The ordering for how to sort the frame.
        min_r (int):
            The smallest r (layer) value to be used to order cluster_ids.

    Returns: (None)
    """
    low_r = frame[frame.r == min_r].sort_values(order)
    id2i  = dict((_id, i) for i, _id in enumerate(low_r.cluster_id))
    frame.cluster_id = frame.cluster_id.map(id2i)


def _padded_train_and_target(train: Train,
                             target: Target)\
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
    p_train  = pad_sequences(train, padding="post", value=0, dtype=np.float32)
    p_target = pad_sequences(target, padding="post", value=0, dtype=np.int32)

    # The final column in target is the "padding" category. For each padding
    # that we added, assign the column in its row to 1.
    for matrix in p_target:
        for row in matrix:
            if np.sum(row) == 0:
                row[-1] = 1
    return p_train, p_target
