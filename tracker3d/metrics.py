""" tracker3d/metrics.py

This file contains functions useful for measuring how well a keras model
performs at predicting results for the tracking problem.

Author: Daniel Zurawski
Author: Keshav Kapoor
Organization: Fermilab
Grammar: Python 3.6.1
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from .tracker_types import PMatrix, Target, Train, Event
from . import utils


def wrong_classifications(guess: PMatrix,
                          target: PMatrix,
                          verbose: bool=False)\
        -> int:
    """ Get the number of incorrect classifications.

    Arguments:
        target (PMatrix):
            The ground truth probability matrix.
        guess (PMatrix):
            The prediction probability matrix.
        verbose (bool):
            True if you want this function to print things.
            False if you want this function to keep its big mouth shut.

    Returns (int):
        The number of incorrect classifications.
    """
    wrong_category = 0  # The number of wrong classifications.

    # Iterate through the guess rows and find wrong classifications.
    for j, row in enumerate(guess):
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
    return utils.to_categorical(utils.from_categorical(probs), probs.shape[1])


def threshold_probabilities(guess: PMatrix,
                            threshold: float)\
        -> PMatrix:
    """ Return a threshold matrix from *guess*.

    The return matrix is the guess matrix, except all values greater than
    or equal to *threshold* are 1'd and values less than *threshold* are 0'd.

    If threshold is 0.4:
    guess:           return:
    [[0.4, 0.5,  0.1],     [[1, 1, 0],
     [0.2, 0.2,  0.6],  ->  [0, 0, 1],
     [0.4, 0.4,  0.2]]      [1, 1, 0],
     [0.3, 0.35, 0.35]      [0, 0, 0]]

    Arguments:
        guess (PMatrix):
            The prediction matrix.
        threshold (float):
            A threshold value.

    Return: PMatrix
    """
    matrix = np.copy(guess)
    matrix[guess < threshold]  = 0
    matrix[guess >= threshold] = 1
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


def discrete_accuracy(guess: PMatrix,
                      target: PMatrix,
                      padding: bool=True)\
        ->float:
    """ Return the discrete accuracy for this guess.

    The discrete accuracy is how measured as what percent of rows in
    *guess* have the largest probability at the same index as the
    1 value within the target matrix's row.

    Arguments:
        guess (PMatrix):
            A matrix of track probability predictions for *train* event.
        target (PMatrix):
            A matrix of ground truth probabilities for *train* event.
        padding (bool):
            True if padding should be removed.
            False if padding should be accounted for within this calculation.

    Returns: (float):
        The discrete accuracy
    """
    if padding:
        guess      = utils.remove_padding_matrix(guess, target)
        target     = utils.remove_padding_matrix(target, target)
    discrete = discretize(guess)
    return np.equal(target, discrete).all(axis=1).sum() / target.shape[0]


def discrete_accuracy_all(train: Train,
                          guesses: Target,
                          targets: Target,
                          padding: bool=True)\
        -> float:
    """ Return the average discrete accuracy.

    The discrete accuracy is how measured as what percent of rows in a
    prediction have the largest probability at the same index as the
    1 value within the target matrix's row.

    Arguments:
        train (Train):
            A list of training Events.
        guesses (Target):
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
        discrete = discretize(guesses[i])  # Discrete predictions matrix.
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


def threshold_metrics(guess: PMatrix,
                      target: PMatrix,
                      threshold: float,
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
        guess (PMatrix):
            A prediction matrix.
        threshold (float):
            A threshold value such that any value lower than this in guess
            matrix is 0'd and any value higher or equal to this is 1'd.
        padding (bool):
            If padding is true, then the last column in target and guess
            is not considered. This is because the last column is the padding
            column by convention.

    Returns: np.ndarray
    """
    # Remove the padding column, if necessary.
    if padding:
        tg_matrix = utils.remove_padding_matrix(target, target)
        pd_matrix = utils.remove_padding_matrix(guess, target)
    else:
        tg_matrix = target
        pd_matrix = guess
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


def track_metrics(guess: PMatrix,
                  target: PMatrix,
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
    x = (number of columns in target and prediction with the same index
            that contain the same element at least *threshold* percent of the
            time.
    y = (number of tracks that aren't padding)
    Return: (x / y) -> (2 / 2) -> (100%)

    Arguments:
        target (PMatrix):
            The target matrix.
        guess (PMatrix):
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
    guess    = guess[:, :-1][np.any(nopad != 0, axis=1)] if padding else guess
    n_tracks = target.shape[1]
    discrete = discretize(guess)
    equals   = np.equal(target, discrete).transpose()
    rights   = np.sum([target[i].sum() > 0 and threshold * len(e) <= e.sum()
                       for i, e in enumerate(equals)])
    return rights / n_tracks


def accuracy_vs_tracks_boxplot(guesses: Target,
                               targets: Target,
                               noise: bool=True,
                               padding: bool=True,
                               plot=True)\
        -> Tuple[np.ndarray, np.ndarray]:
    """ Plot and return accuracy vs number of tracks.

    Arguments:
        guesses:
            An array of prediction matrices.
        targets:
            An array of target matrices.
        noise:
            True if the matrices have noise column.
        padding:
            True if the matrices have a padding column.
        plot:
            True if you want this function to provide a box plot.
            If true, then "n" box and whisker plots are drawn, where "n" is the
            number of tracks represented in the target matrix. Each box plot
            corresponds to the discrete accuracies of all events with that
            number of tracks. So, for example, the 6th box and whisker plot
            will correspond to the discrete accuracies of all events with 6
            tracks.

    Returns: Tuple[np.ndarray, np.ndarray]
        
    """
    tracks = [utils.number_of_tracks(matrix, padding, noise)
              for matrix in targets]
    acc    = [discrete_accuracy(guesses[i], targets[i], padding)
              for i in range(targets.shape[0])]
    if plot:
        boxes = [[] for _ in range(targets.shape[2] + 1)]
        for i in range(targets.shape[0]):
            boxes[tracks[i]].append(acc[i])
        plt.figure()
        plt.xlabel("Number of Tracks")
        plt.ylabel("Discrete Accuracy")
        plt.title("Accuracy vs # of Tracks")
        plt.boxplot(boxes, showfliers=False)
        plt.xticks([i for i in range(len(boxes))], [i-1 for i in range(len(boxes))])

        plt.show()
    return tracks, acc


def hit_accuracy(hit_num: int,
                 predictions: Target,
                 targets: Target,
                 threshold_value: float,
                 threshold: bool=True)\
        -> float:
    """ % of events with a given hit classified correctly.

    Arguments:
        hit_num (int):
            The hit number to check accuracy for.
        targets (Target):
            A list of ground truth probability matrices.
        predictions (Target):
            A list of predictions probability matrices.
        thresholdValue (Float):
            The threshold value(confidence) to classify the hits in tracks.
        threshold (bool):
            A boolean to check if we want to use threshold or discrete matrices.

    Returns: % of events with a given hit classified correctly.
    """

    # Discretizes or makes a threshold matrix and removes padding from the prediction matrices.
    prob = []

    # Check if we want threshold or discrete accuracy
    if threshold:
        for i, event in enumerate(predictions):
            prob.append(threshold_probabilities(event, threshold_value))

    else:
        for i, event in enumerate(predictions):
            prob.append(discretize(event))

    probability = np.array(prob)
    prob_no_padding = []
    for i, event in enumerate(probability):
        prob_no_padding.append(utils.remove_padding_matrix(event, targets[i]))
    prob_no_padding = np.array(prob_no_padding)

    # Removes the padding from the Target matrices.
    target_no_padding = []
    for i, target in enumerate(targets):
        target_no_padding.append(utils.remove_padding_matrix(target, target))
    target_no_padding = np.array(target_no_padding)

    # Counts the number of events that classified the given hit correctly.
    accuracy = 0
    count = 0

    for i, target in enumerate(target_no_padding):
        if hit_num <= target.shape[0]:  # Checks if the hit exists for the event
            count += 1
            if prob_no_padding[i][hit_num][np.argmax(target[hit_num])] == 1:
                accuracy += 1

    return accuracy/count


def threshold_boxplot(guesses: Target,
                      targets: Target,
                      thresholds: List[float],
                      variation: str,
                      padding: bool=True,
                      plot=True)\
        -> np.ndarray:
    """ Create multiple box plots for each threshold in thresholds.

    Arguments:
        guesses
        targets
        thresholds
        padding
        variation
        plot

    Returns (np.ndarray)
        accuracies
    """
    variations = ("correct", "incorrect", "many", "none")
    if variation.lower() not in variations:
        print("Error: the 'variation' variable was not found in function:")
        print("'threshold_boxplot' in the 'metrics' module.")
        return np.array([])
    index = variations.index(variation.lower())

    accuracies = [[threshold_metrics(guesses[i], targets[i], th, padding)
                   for i in range(targets.shape[0])]
                  for th in thresholds]
    accuracies = np.array(accuracies).transpose()[index]

    if plot:
        plt.figure()
        plt.boxplot(accuracies, showfliers=False)
        plt.xlabel("Threshold Value")
        plt.ylabel("Probability")
        if variation == "correct":
            plt.title("Prob[Hit assigned correctly with at least threshold certainty]")
        elif variation == "incorrect":
            plt.title("Prob[Hit assigned incorrectly with at least threshold certainty]")
        elif variation == "many":
            plt.title("Prob[Hit assigned to multiple tracks with at least threshold certainty]")
        elif variation == "none":
            plt.title("Prob[Hit assigned to no track with at least threshold certainty]")
        plt.xticks(range(1, 1 + len(thresholds)), thresholds)
        plt.show()
    return accuracies


def avg_accuracy_vs_track_scatter(guesses: Target,
                                  targets: Target,
                                  noise: bool=True,
                                  padding: bool=True,
                                  plot=True)\
        -> Tuple[np.ndarray, np.ndarray]:
    tracks, accuracy = accuracy_vs_tracks_boxplot(guesses, targets, noise,
                                                  padding, plot=False)
    num_tracks = int(np.amax(tracks))
    out_accuracy = np.zeros(num_tracks)
    for i in range(num_tracks):
        tot_acc = 0
        count = 0
        for j, track in enumerate(tracks):
            if track == i + 1:  # Checks the track size.
                # Adds all the accuracies of events with the same track size.
                tot_acc = tot_acc + accuracy[j]
                count += 1
        if count > 0:
            # Calculates the average accuracy for the track size.
            out_accuracy[i] = tot_acc / count

    track = np.zeros(num_tracks)

    # Creates the track size array.
    for i in range(num_tracks):
        track[i] = i + 1

    if plot:
        plt.figure()
        plt.scatter(track, out_accuracy)
        plt.show()

    return track, out_accuracy


def number_of_crossings(event: Event,
                        guesses: PMatrix,
                        order: Tuple[str, str, str])\
        -> int:
    # Get list of float pairs such that each pair corresponds to a unique track
    # pair[0] is equal to the phi of the lowest radius hit.
    # pair[1] is equal to the phi of the highest radius hit.
    tracks = [[] for _ in range(guesses.shape[1])]
    cats   = utils.from_categorical(guesses)
    for i, hit in enumerate(event):
        tracks[cats[i]].append(hit)
    r_idx   = order.index("r")
    phi_idx = order.index("phi")
    tracks  = [sorted(track, key=lambda x: x[r_idx]) for track in tracks]
    phis    = [(track[0][phi_idx], track[-1][phi_idx])
               for track in tracks if track]
    phis    = [(phi[0] % (np.pi * 2), phi[1] % (np.pi * 2)) for phi in phis]
    phis    = [(phi[0], phi[1] + (2 * np.pi if phi[0] > phi[1] else 0))
               for phi in phis]

    # Calculate the number of crossings.
    crosses = 0
    for x in range(len(phis)):
        for y in range(x + 1, len(phis)):
            low  = (phis[x][0] < phis[y][0])
            high = (phis[x][1] < phis[y][1])
            crosses += (low != high)
    return crosses
