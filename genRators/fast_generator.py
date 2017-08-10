"""

"""

import numpy as np
import pandas as pd
import random
from typing import Sequence


def make_bank(source: pd.DataFrame,
              number_of_tracks: int) \
        -> pd.DataFrame:
    """
    """
    groups = source.groupby(["event_id", "cluster_id"])
    tracks = [track for (_, track) in groups if len(track) > 1]
    sample = random.choices(population=tracks, k=number_of_tracks)
    bank : pd.DataFrame = pd.concat(sample)  # pycharm no type check pd.concat.
    return bank


def generate(logistics: Sequence[Sequence[int]],
             bank: pd.DataFrame)\
        -> pd.DataFrame:
    """
    Arguments:
        logistics: Sequence[Sequence[int, int]]
            A sequence of integer pairs.
            Pairs represent information used to specify event generation.
            Pair[0] -> Number of events.
            Pair[1] -> Tracks per event.
            Example: logistics = [(3, 5), (10, 2), (6, 12)] translates to...
            "Return 1 pd.DataFrame with:
              3 events with  5 tracks each,
             10 events with  2 tracks each,
              6 events with 12 tracks each.".
        bank: pd.DataFrame
            The bank of tracks to sample tracks from.

    Returns:
        A new pd.DataFrame consisting of generated events.
    """
    generated = []  # Generated tracks go here.
    groups    = bank.groupby(["event_id", "cluster_id"])
    tracks    = [track for (_, track) in groups if len(track) > 1]
    event_id  = 0  # Increments every time we add an event to "generated".
    for number_of_events, tracks_per_event in logistics:
        for _ in range(number_of_events):
            samples = random.sample(population=tracks, k=tracks_per_event)
            for i, track in enumerate(samples):
                # Randomly rotate track about origin.
                phi = np.random.uniform(-np.pi, np.pi)
                rot = np.array([[np.cos(phi), -np.sin(phi)],
                                 [np.sin(phi), np.cos(phi)]])
                coordinates = np.array([track["x"], track["y"]]).T @ rot
                track["cluster_id"] = i
                track["event_id"]   = event_id
                track["x"]          = coordinates[:, 0]
                track["y"]          = coordinates[:, 1]
                generated.append(track)
            event_id += 1
            print("\rGenerated {}".format(event_id), end="")
    print("")
    frame: pd.DataFrame = pd.concat(generated)
    frame: pd.DataFrame = frame[["event_id", "cluster_id", "x", "y"]]
    return frame.sample(frac=1).drop_duplicates()

if __name__ == '__main__':
    print("Starting program.")
    ramp = pd.read_csv("../datasets/raw/ramp.csv")
    dist = [(200, i) for i in range(1, 1 + 22)]
    gen  = generate(logistics=dist, bank=ramp)
    gen.to_csv("../datasets/raw/generated.csv")
    print("Ending program.")
