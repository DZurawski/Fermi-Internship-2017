""" tracker/metrics.py

Author: Daniel Zurawski
Organization: Fermilab
Grammar: Python 3.6.1
"""

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
from . import extractor as ext, utils


def number_of_hits(
        frame : pd.DataFrame,
        ) -> int:
    """ Get the number of hits.
    :param frame:
        A pd.DataFrame
    :return:
        The number of hits within this "frame". Padding does not count as a
        valid hit, but noisy hits do count.
    """
    if {"padding"}.issubset(frame.columns):
        return len(utils.remove_padding(frame))
    else:
        return len(frame)


def number_of_tracks(
        frame : pd.DataFrame,
        noise_counts_as_a_track : bool = False,
        ) -> int:
    """ Get the number of tracks.
    :param frame:
        A pd.DataFrame
    :param noise_counts_as_a_track:
        If True, then noise counts as a track.
    :return:
        The number of tracks within this "frame". Padding does not count as a
        track. Noise may or may not count as a track.
    """
    if {"padding"}.issubset(frame.columns):
        frame = utils.remove_padding(frame)
    if {"noise"}.issubset(frame.columns) and not noise_counts_as_a_track:
        frame = utils.remove_noise(frame)
    return frame.groupby(["event_id", "cluster_id"]).size().max()



def number_of_crossings(
        frame : pd.DataFrame,
        order : List[str],
        guess : Optional[np.ndarray] = None,
        ) -> float:
    # TODO
    return -1


def distributions(
        frame : pd.DataFrame,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ Return unique number of tracks per event and number of occurrences.
    :param frame:
        A pd.DataFrame.
    :return:
        A tuple of two np.ndarrays. The first np.ndarray contains a sorted
        array of the unique number of tracks within an event. The second
        np.ndarray contains an array of the number of events with that
        unique number of tracks.
        Example output: np.array(2, 5, 6), np.array(5, 4, 1)
        Translates to:
        (5 events with 2 tracks),
        (4 events with 5 tracks),
        (6 tracks with 1 track)
    """
    frame  = utils.remove_noise(utils.remove_padding(frame))
    events = utils.list_of_groups(frame, group="event_id")
    sizes  = np.array([event["cluster_id"].max() + 1 for event in events])
    return tuple(np.unique(sizes, return_counts=True))


def discrete(
        matrix : np.ndarray
        ) -> np.ndarray:
    """ Get a probability matrix of discrete (0 or 1) values.
    :param matrix:
        The np.ndarray that will be made discrete.
    :return:
        A discrete np.ndarray. All values are either 0 or 1. All rows
        add up to exactly 1.
    """
    return utils.to_categorical(utils.from_categorical(matrix))


def threshold(
        matrix : np.ndarray,
        thresh : float
        ) -> np.ndarray:
    """ Get a discrete matrix such that values above threshold are 1'd. """
    threshold_matrix = np.copy(matrix)
    threshold_matrix[thresh >  matrix] = 0
    threshold_matrix[thresh <= matrix] = 1
    return threshold_matrix


def percent_of_hits_assigned_correctly(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : Union[np.ndarray, List[np.ndarray]],
        order   : List[str],
        do_not_factor_in_padding : bool = True,
        do_not_factor_in_noise   : bool = False,
        ) -> float:
    """ Get the percent of hits that were assigned to the correct track. """
    if isinstance(frames, pd.DataFrame):
        return percent_of_hits_assigned_correctly(
                [frames], [guesses], order,
                do_not_factor_in_padding, do_not_factor_in_noise)
    n_hits, n_correct = 0, 0
    for i, guess in enumerate(guesses):
        guess = utils.from_categorical(guess)
        frame = frames[i]
        if do_not_factor_in_padding:
            guess  = utils.remove_padding(frame, guess, order)
            frame  = utils.remove_padding(frame)
        if do_not_factor_in_noise:
            guess  = utils.remove_noise(frame, guess, order)
            frame  = utils.remove_noise(frame)
        target = ext.extract_output(frame, order, categorical=False)
        n_correct += np.equal(guess, target).sum()
        n_hits    += len(guess)
    return n_correct / n_hits


def percent_of_events_with_correct_number_of_tracks(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : List[np.ndarray],
        order   : List[str],
        ) -> float:
    if isinstance(frames, pd.DataFrame):
        return percent_of_tracks_assigned_correctly(
                utils.list_of_groups(frames, "event_id"), guesses, order)
    n_correct = 0
    for i in range(len(frames)):
        guess  = utils.from_categorical(guesses[i])
        target = ext.extract_output(frames[i], order, categorical=False)
        n_correct = len(np.unique(guess)) == len(np.unique(target))
    return n_correct / len(frames)


def percent_of_tracks_assigned_correctly(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : Union[np.ndarray, List[np.ndarray]],
        order   : List[str],
        percent : float = 1.0,
        do_not_factor_in_padding : bool = True,
        do_not_factor_in_noise   : bool = False,
        ) -> float:
    """ Get the percent of tracks that were assigned the correct hits. """
    if isinstance(frames, pd.DataFrame):
        return percent_of_tracks_assigned_correctly(
                [frames], [guesses], order, percent,
                do_not_factor_in_padding, do_not_factor_in_noise)
    n_tracks, n_correct = 0, 0
    for i, guess in enumerate(guesses):
        guess  = discrete(guess)
        frame  = frames[i]
        if do_not_factor_in_padding:
            guess = utils.remove_padding(frame, guess, order)
            frame = utils.remove_padding(frame)
        if do_not_factor_in_noise:
            guess = utils.remove_noise(frame, guess, order)
            frame = utils.remove_noise(frame)
        target = ext.extract_output(frame, order).transpose()
        guess  = guess.transpose()
        for r in range(len(target)):
            track = 0
            for c in range(len(target[r])):
                track += (target[r, c] == 1 and guess[r, c] == 1)
            n_correct += (percent <= (track / len(target[r])))
        n_tracks += len(guess)
    return n_correct / n_tracks


def threshold_metrics(
        frame  : pd.DataFrame,
        guess  : np.ndarray,
        thresh : float,
        order  : List[str],
        ) -> np.ndarray:
    """ Return metrics corresponding to the threshold matrix.
        Returns a np.ndarray consisting of four floats.
        float 0: The probability that a hit was assigned to a correct track
            by the threshold matrix.
        float 1: The probability that a hit was assigned to an incorrect track
            by the threshold matrix.
        float 2: The probability that a hit was assigned to more than 1 track
            by the threshold matrix.
        float 3: The probability that a hit was assigned to no track by the
            threshold matrix.
    """
    # Remove the padding column, if necessary.
    guess  = utils.remove_padding(frame, guess, order)
    frame  = utils.remove_padding(frame)
    target = ext.extract_output(frame, order)
    n_hits = number_of_hits(frame)
    matrix = threshold(guess, thresh)
    stack  = np.dstack((target, matrix)).transpose((0, 2, 1))
    rights = np.sum([pair[1, np.argmax(pair[0])] == 1 for pair in stack])
    wrongs = np.sum((stack[:, 0] - stack[:, 1] < 0).any(axis=1))
    multi  = np.sum(np.sum(matrix, axis=1) > 1)  # Hits assigned to multiple.
    no_tks = np.sum(np.sum(matrix, axis=1) < 1)  # Hits unassigned to any.
    return np.array([rights, wrongs, multi, no_tks]) / n_hits


def accuracy_vs_tracks(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : List[np.ndarray],
        order   : List[str],
        ) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(frames, pd.DataFrame):
        groups = utils.list_of_groups(frames, "event_id")
        return accuracy_vs_tracks(groups, guesses, order)
    n_tracks = [number_of_tracks(frame) for frame in frames]
    accuracy = [percent_of_hits_assigned_correctly(frames[i], guesses[i], order)
                for i in range(len(guesses))]
    return np.array(n_tracks), np.array(accuracy)


def accuracy_vs_thresholds(
        frames   : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses  : List[np.ndarray],
        order    : List[str],
        threshes : List[float],
        mode     : str = "correct",  # ("correct", "incorrect", "many", "none")
        ) -> Tuple[np.ndarray, np.ndarray]:
    mode  = mode.lower()
    modes = ("correct", "incorrect", "many", "none")
    if mode not in modes:
        print("Error: the 'variation' variable was not found in function:")
        return np.ndarray([]), np.ndarray([])
    if isinstance(frames, pd.DataFrame):
        groups = utils.list_of_groups(frames, "event_id")
        return accuracy_vs_thresholds(groups, guesses, order, threshes, mode)
    index = modes.index(mode)
    accuracy = [[threshold_metrics(frames[i], guesses[i], thresh, order)
                 for i in range(len(frames))] for thresh in threshes]
    return np.array(threshes), np.array(accuracy).transpose()[index]
