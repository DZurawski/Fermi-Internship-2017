""" tracker3d/metrics.py

This file contains functions useful for measuring how well a keras model
performs at predicting results for the tracking problem.

Author: Daniel Zurawski
Author: Keshav Kapoor
Organization: Fermilab
Grammar: Python 3.6.1
"""

from .utils import to_categorical, from_categorical, remove_padding_matrix
from .utils import plot3d, display_side_by_side, remove_padding_event
import numpy as np
from .tracker_types import PMatrix, Target, Train, Event
from typing import Any

# Keras tensors are different types depending on backend used.
# For instance, tensorflow uses a different type than theano.
Tensor = Any


def wrong_classifications(target: PMatrix,
                          prediction: PMatrix,
                          verbose: bool=False)\
        -> int:
    """ Get the number of incorrect classifications.

    Arguments:
        target (PMatrix):
            The ground truth probability matrix.
        prediction (PMatrix):
            The prediction probability matrix.
        verbose (bool):
            True if you want this function to print things.
            False if you want this function to keep its big mouth shut.

    Returns (int):
        The number of incorrect classifications.
    """
    wrong_category = 0  # The number of wrong classifications.

    # Iterate through the prediction rows and find wrong classifications.
    for j, row in enumerate(prediction):
        predict = np.argmax(row)  # The prediction for this row.
        answer  = np.argmax(target[j])  # The correct answer.
        if predict != answer:
            if verbose:
                print("The model predicted hit {} incorrectly."
                      .format(j))
                print("Certainty of prediction: {}."
                      .format(row[predict]))
                print("The correct track was {}."
                      .format(answer if answer < target.shape[1] else "noise"))
            wrong_category += 1  # We found a wrong classificaiton. Increment!
    if verbose:
        print("-- There were {} wrong classifications."
              .format(wrong_category))
    return wrong_category


def discretize(probs: PMatrix)\
        -> PMatrix:
    """ Return a discrete probability matrix.

    A discrete probability matrix is derived from a regular probability
    matrix. Each row in the discrete matrix contain a single 1 and the rest
    0's. The column that this 1 is placed in corresponds to the maximum
    probability within the regular probability matrix's row. In the case of a
    tie, choose the column with lowest index.

    probs:                return:
    [[0.3, 0.5, 0.2],     [[0, 1, 0],
     [0.2, 0.2, 0.6],  ->  [0, 0, 1],
     [0.4, 0.4, 0.2]]      [1, 0, 0]]

    Arguments:
        probs (PMatrix):
            A matrix consisting of positive entries and where each row
            adds up to exactly 1.

    Returns (PMatrix):
        Return a matrix of the same shape as probabilities such that each
        row contains a single 1 and the rest 0's.
    """
    return to_categorical(from_categorical(probs), probs.shape[1])


def threshold_probabilities(prediction: PMatrix,
                            threshold: float)\
        -> PMatrix:
    """ Return a threshold matrix from *prediction*.

    The return matrix is the prediction matrix, except all values greater than
    or equal to *threshold* are 1'd and values less than *threshold* are 0'd.

    If threshold is 0.4:
    prediction:           return:
    [[0.4, 0.5,  0.1],     [[1, 1, 0],
     [0.2, 0.2,  0.6],  ->  [0, 0, 1],
     [0.4, 0.4,  0.2]]      [1, 1, 0],
     [0.3, 0.35, 0.35]      [0, 0, 0]]

    Arguments:
        prediction (PMatrix):
            The prediction matrix.
        threshold (float):
            A threshold value.

    Return: PMatrix
    """
    matrix = np.copy(prediction)
    matrix[prediction < threshold]  = 0
    matrix[prediction >= threshold] = 1
    return matrix


def hits_per_track(matrix: PMatrix,
                   verbose: bool=False)\
        -> np.ndarray:
    """ Return an array of integers representing the number of hits per track.

    Arguments:
        matrix (PMatrix):
            A probability matrix.
        verbose (bool):
            True if this functions should print anything. Else, no printing.

    Returns (np.ndarray):
        An array of integers. The cell at index *i* contains the number of
        hits that are a member of track *i* as defined by *probs*.
    """
    hits = discretize(matrix).sum(axis=0)
    if verbose:
        for i, hit in enumerate(hits):
            print("Hits in Track {0}: {1}".format(i, hit))
    return hits


def probability_hits_per_track(targets: Target,
                               track_id: int,
                               num_hits: int,
                               verbose: bool=False)\
        -> float:
    """ Determine how well predictions are for a specific track id.

    Given a list of probability matrices *target*, a track id *track_id* and
    an expected number of hits per track *num_hits*, provide the percent of
    matrices that correctly assign this track id the correct number of hits.

    Arguments:
        targets (Target):
            A list of probability matrices.
        track_id (int):
            The id of the track.
        num_hits (int):
            The proper number of hits that a track is expected to have.
        verbose:
            If True, will print out the number of matrices that correctly
            assign the correct number of hits to the track with *track_id*.
            Else, prints nothing.

    Returns: (float)
    The percent of matrices that correctly assign *num_hits* number of hits to
    the track with track id, *track_id*.
    """
    answer = sum([(hits_per_track(m)[track_id] == num_hits) for m in targets])
    if verbose:
        print("Number correct: {}".format(answer))
    return answer / len(targets)  # Percent of events with correct hits/track.


def discrete_accuracy(event: Event,
                      target: PMatrix,
                      prediction: PMatrix,
                      verbose: bool=False,
                      padding: bool=True)\
        ->float:
    """ Return the discrete accuracy for this prediction.

    The discrete accuracy is how measured as what percent of rows in
    *prediction* have the largest probability at the same index as the
    1 value within the target matrix's row.

    Arguments:
        event (Event):
            An array of hit coordinates.
        prediction (PMatrix):
            A matrix of track probability predictions for *train* event.
        target (PMatrix):
            A matrix of ground truth probabilities for *train* event.
        verbose (bool):
            True if graphs and tables should both be displayed.
            False if nothing will be displayed.
        padding (bool):
            True if padding should be removed.
            False if padding should be accounted for within this calculation.

    Returns: (float):
        The discrete accuracy
    """
    if padding:
        event      = remove_padding_event(event)
        prediction = remove_padding_matrix(prediction)
        target     = remove_padding_matrix(target)

    display_plots_and_tables = False  # If anything goes wrong, flips to True.
    discrete = discretize(prediction)

    # Time to calculate the accuracy
    accuracy = 0
    for i, row in enumerate(discrete):
        # Check if the 1 in a row in the discrete accuracy matrix is at the
        # same index as the 1 in a row for the target matrix.
        equivalent = np.equal(np.argmax(row), np.argmax(target[i]))
        accuracy  += int(equivalent)
        if verbose and not equivalent:
            print("Wrong hit in row: {}".format(i))
            display_plots_and_tables = True
    percent = accuracy / len(target)  # The percent correct.
    if verbose:
        print("The accuracy is: {}".format(percent))
        if display_plots_and_tables:
            plot3d(event, prediction, target)
            display_side_by_side(event, prediction, target)
    return percent


def discrete_accuracy_all(train: Train,
                          targets: Target,
                          predictions: Target,
                          padding: bool=True)\
        -> float:
    """ Return the average discrete accuracy.

    The discrete accuracy is how measured as what percent of rows in a
    prediction have the largest probability at the same index as the
    1 value within the target matrix's row.

    Arguments:
        train (Train):
            A list of training Events.
        predictions (Target):
            A list of predicted probability matrices.
        targets (Target):
            A list of ground truth probability matrices.
        padding (bool):
            True if padding should be removed.
            False if padding should be accounted for within this calculation.

    Returns: (float)
        Average discrete accuracy
    """
    zero     = np.zeros(train.shape[2])  # Used to determine row as padding.
    accuracy = 0
    count    = 0
    for i, event in enumerate(train):
        discrete = discretize(predictions[i])  # Discrete predictions matrix.
        for j, hit in enumerate(event):
            if padding and np.all(np.equal(hit, zero)):
                # Skip padding events.
                continue
            else:
                # Check if a row in the discrete accuracy matrix and the
                # target matrix are equal. If so, increment accuracy.
                accuracy += np.all(np.equal(discrete[j], targets[i][j]))
                count    += 1
    return accuracy / count  # The percent of correct hits, not including noise.


def threshold_metrics(target: PMatrix,
                      prediction: PMatrix,
                      threshold: int,
                      padding: bool=True)\
        -> np.ndarray:
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

    Arguments:
        target (PMatrix):
            The target matrix.
        prediction (PMatrix):
            A prediction matrix.
        threshold (float):
            A threshold value such that any value lower than this in prediction
            matrix is 0'd and any value higher or equal to this is 1'd.
        padding (bool):
            If padding is true, then the last column in target and prediction
            is not considered. This is because the last column is the padding
            column by convention.

    Returns: np.ndarray
    """
    # Remove the padding column, if necessary.
    tg_matrix = remove_padding_matrix(target) if padding else target
    pd_matrix = remove_padding_matrix(prediction) if padding else prediction
    n_hits    = tg_matrix.shape[0]
    th_matrix = threshold_probabilities(pd_matrix, threshold)  # Threshold mtx.
    stack     = np.dstack((tg_matrix, th_matrix)).transpose((0, 2, 1))

    # For each pair of rows from tg_matrix and th_matrix, see if the element
    # at the index where tg_matrix equals 1 also equals 1.
    rights = np.sum([pair[1, np.argmax(pair[0])] == 1 for pair in stack])

    # For each row in tg_matrix and th_matrix, Subtract the th_matrix row from
    # the tg_matrix row. If that resulting 1-D array has any negative values,
    # then we know that that there was some index such that the tg_matrix row
    # did not categorize the hit as this index's track, but the th_matrix
    # did, in fact, categorize the hit as this index's track.
    wrongs = np.sum((stack[:, 0] - stack[:, 1] < 0).any(axis=1))

    # If a row within the th_matrix has a sum of greater than 1, then we know
    # that the th_matrix classified this hit with more than 1 track.
    multis = np.sum(np.sum(th_matrix, axis=1) > 1)  # Hits assigned to multiple.

    # If a row within the th_matrix has a sum of less than 1, then we know that
    # the th_matrix did not classify this hit with any track.
    notrks = np.sum(np.sum(th_matrix, axis=1) < 1)  # Hits unassigned to any.

    return np.array([rights, wrongs, multis, notrks]) / n_hits


def track_metrics(target: PMatrix,
                  pred: PMatrix,
                  threshold: int,
                  padding=True)\
        -> np.ndarray:
    """ % of tracks with at least *threshold* % of correct hits assigned.

    So, suppose *threshold* were 0.5 and we had matrices with padding as the
    last column.
    Target                    Prediction
    [[0, 1, 0],    [[0, 1],   [[0.4, 0.6, 0.0],     [[0, 1],
     [1, 0, 0], --> [1, 0],    [0.9, 0.1, 0.0], -->  [1, 0],
     [1, 0, 0],     [1, 0]]    [0.3, 0.7, 0.0],      [0, 1]]
     [0, 0, 1]]                [0.0, 0.0, 1.0]]
    Results to:
    x = (2 == number of columns in target and prediction with the same index
            that contain the same element at least *threshold* percent of the
            time.
    y = (2 == number of tracks that aren't padding)
    Return: x / y == 2 / 2 = 100%

    Arguments:
        target (PMatrix):
            The target matrix.
        pred (PMatrix):
            The prediction matrix.
        threshold (int):
            The threshold value between 0 and 1.
        padding (bool):
            If True, last column and any padding rows are not considered, since
            they correspond to padding.

    Returns: (np.ndarray)
    """
    nopad    = target[:, :-1]
    target   = nopad[np.any(nopad != 0, axis=1)] if padding else target
    pred     = pred[:, :-1][np.any(nopad != 0, axis=1)] if padding else pred
    n_tracks = target.shape[1]
    discrete = discretize(pred)
    equals   = np.equal(target, discrete).transpose()
    rights   = np.sum([target[i].sum() > 0 and threshold * len(e) <= e.sum()
                       for i, e in enumerate(equals)])
    return rights / n_tracks
