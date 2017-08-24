""" extractor.py/extractor
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Any
from . import utils, metrics


def extract_input(
        frame   : Union[pd.DataFrame, List[pd.DataFrame]],
        order   : List[str],
        nan_to  : Any = 0,
        ) -> np.ndarray:
    """ Extract a model input array from a frame.
    :param frame:
        A pd.DataFrame to extract from. It can have multiple event_ids or can
        be a list of pd.DataFrames.
    :param order:
        A permutation of ["phi", "r", "z"] that will be used to sort the input.
    :param nan_to:
        What value should NaN values be transformed into.
    :return:
        If there is a single event within the frame, return a 2D matrix
        consisting of input data. Input data is an array of [phi, r, z]
        position information. Thus, it has shape (None, 3).
        If there are multiple events or a list of frames was passed into
        "frame", return a cube of data such that each depth consists of a
        2D matrix of input data corresponding to each event.
    """
    if isinstance(frame, pd.DataFrame):
        if len(pd.unique(frame["event_id"])) > 1:
            groups = utils.list_of_groups(frame, "event_id")
            return extract_input(groups, order, nan_to)
        else:
            return frame.sort_values(order).fillna(nan_to)[order].get_values()
    else:
        return np.array([extract_input(e, order, nan_to) for e in frame])


def extract_output(
        frame       : Union[pd.DataFrame, List[pd.DataFrame]],
        order       : List[str],
        column      : str = "cluster_id",
        categorical : bool = True,
        ) -> np.ndarray:
    """ Extract a model output array from a frame.
    :param frame:
        A pd.DataFrame to extract from. It can have multiple event_ids or can
        be a list of pd.DataFrames.
    :param order:
        A permutation of ["phi", "r", "z"] that will be used to sort the input.
    :param column:
        The name of the column to extract the output from.
    :param categorical:
        True if the output should be a categorical probability matrix.
        False if the output should be a list of categories.
    :return:
        If "categorical" is True:
        If there is a single event within the frame, return a 2D matrix
        consisting of a probability target matrix.
        Thus, it has shape (None, *number of categories in cluster_id*).
        If there are multiple events or a list of frames was passed into
        "frame", return a cube of data such that each depth consists of a
        2D matrix of input data corresponding to each event.
        If "categorical" is False:
        Return a list of such that the i'th hit belongs to the i'th category
        in the list.
        """
    if isinstance(frame, pd.DataFrame):
        if len(pd.unique(frame["event_id"])) > 1:
            groups = utils.list_of_groups(frame, "event_id")
            return extract_output(groups, order, column, categorical)
        sort = frame.sort_values(order)[column].get_values()
        return utils.to_categorical(sort) if categorical else sort
    else:
        return np.array([extract_output(e, order, column, categorical)
                        for e in frame])


def input_output_generator(
        events : List[pd.DataFrame],
        batch  : int,
        order  : List[str],
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ A input-output generator for a model.
    :param events:
        A list of pd.DataFrames to get input-output from.
    :param batch:
        How many instances of input-output to output per yield.
    :param order:
        A permutation of ["phi", "r", "z"]
    :return:
        Yields a tuple of input-output data.
    """
    i = 0
    while True:
        in_  = [extract_input(event, order) for event in events[i:i+batch]]
        out_ = [extract_output(event, order) for event in events[i:i+batch]]
        i    = (i + batch) if (i + batch < len(events)) else 0
        yield (np.array(in_), np.array(out_))


def reindex_event_ids(
        frame : pd.DataFrame
        ) -> None:
    frame = frame.copy()
    ids   = frame["event_id"].sort_values()
    dic   = dict((event_id, i) for i, event_id in enumerate(ids))
    frame["event_id"] = frame["event_id"].map(dic)


def reindex_cluster_ids(
        frame : pd.DataFrame
        ) -> None:
    events = utils.list_of_groups(frame, group="event_id")
    for event in events:
        idx  = event.groupby("cluster_id")["r"].transform(min) == event["r"]
        lows = event[idx].sort_values(["phi", "z", "r"])
        dic  = dict((id_, i) for i, id_ in enumerate(lows["cluster_id"]))
        event["cluster_id"] = event["cluster_id"].map(dic)
    return np.concat(events)


def prepare_frame(
        frame    : pd.DataFrame,
        n_rows   : int = -1,
        n_tracks : int = -1,
        n_noise  : int = 0,
        ) -> pd.DataFrame:
    """ Prepare a data set.
    :param frame:
        A pd.DataFrame
    :param n_rows:
        The number of rows that the frame should have after including padding
        and noise.
    :param n_tracks:
        The number of tracks that the frame.
    :param n_noise:
        The number of noise hits to add per event.
    :return:
        The prepared pd.DataFrame such that each event has padding and noise.
    """
    events   = utils.list_of_groups(frame, group="event_id")
    cleans   = []  # The prepared events go in here.
    n_tracks = metrics.number_of_tracks(frame) if n_tracks < 0 else n_tracks
    n_rows   = metrics.number_of_hits(frame) + n_noise if n_rows < 0 else n_rows
    for event_id, event in enumerate(events):
        # Map track ids to indices within a probability matrix.
        idx    = event.groupby("cluster_id")["r"].transform(min) == event["r"]
        lows   = event[idx].sort_values(["phi", "z", "r"])
        id2idx = dict((id_, i) for i, id_ in enumerate(lows["cluster_id"]))
        clean  = pd.DataFrame(data={
            "event_id"   : tuple(event_id for _ in range(len(event))),
            "cluster_id" : tuple(event["cluster_id"].map(id2idx)),
            "r"          : tuple(event["r"]),
            "phi"        : tuple(event["phi"]),
            "z"          : tuple(event["z"]),
            "noise"      : tuple([False for _ in range(len(event))]),
            "padding"    : tuple([False for _ in range(len(event))]), })
        print(clean["cluster_id"])
        n_padding = n_rows - len(clean) - n_noise
        cleans.append(make_noise(clean, n_tracks, event_id, n_noise))
        cleans.append(make_padding(n_tracks + 1, event_id, n_padding))
        cleans.append(clean)
    # noinspection PyTypeChecker
    return pd.concat(cleans)


def make_noise(
        frame      : pd.DataFrame,
        cluster_id : int,
        event_id   : int,
        n_noise    : int,
        ) -> pd.DataFrame:
    """ Create noise.
    :param frame:
        A frame to base the noise on.
    :param cluster_id:
        The cluster_id that the noise should have.
    :param event_id:
        The event_id that the noise should have.
    :param n_noise:
        The number of noise hits to generate
    :return:
        A pd.DataFrame consisting of noisy hits.
    """
    min_z, max_z = frame["z"].min(), frame["z"].max()
    return pd.DataFrame(data={
        "event_id"   : tuple([event_id for _ in range(n_noise)]),
        "cluster_id" : tuple([cluster_id for _ in range(n_noise)]),
        "r"          : tuple(np.random.choice(frame["r"].unique(), n_noise)),
        "phi"        : tuple(np.random.uniform(-np.pi, np.pi, n_noise)),
        "z"          : tuple(np.random.uniform(min_z, max_z, n_noise)),
        "noise"      : tuple([True for _ in range(n_noise)]),
        "padding"    : tuple([False for _ in range(n_noise)]), })


def make_padding(
        cluster_id : int,
        event_id   : int,
        n_padding  : int,
        ) -> pd.DataFrame:
    """ Create padding.
    :param cluster_id:
        The cluster_id that each padding should have.
    :param event_id:
        The event_id that each padding should have.
    :param n_padding:
        The number of padding rows to generate.
    :return:
        A pd.DataFrame consisting of the padding rows.
    """
    return pd.DataFrame(data={
        "event_id"   : tuple([event_id for _ in range(n_padding)]),
        "cluster_id" : tuple([cluster_id for _ in range(n_padding)]),
        "r"          : tuple([np.NaN for _ in range(n_padding)]),
        "phi"        : tuple([np.NaN for _ in range(n_padding)]),
        "z"          : tuple([np.NaN for _ in range(n_padding)]),
        "noise"      : tuple([False for _ in range(n_padding)]),
        "padding"    : tuple([True for _ in range(n_padding)]), })
