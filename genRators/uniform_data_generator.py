""" "uniform_data_generator.py"

Author: ???
Organization: Fermilab
Edited by: Daniel Zurawski, Summer 2017

A data generator that takes an input file (default: "public_train_100MeV.csv")
and creates a .csv file containing events with tracks sampled from the input
file.

This file was not written by the Summer 2017 team. Instead, the Summer 2017
team altered it slightly to be more in line with our work.

Changes include:
1. Adapted from Python 2 syntax to Python 3.6.1 syntax.
2. Removed ModelDealer class and import relating to this class.
3. Added a few comments to help explain what the code is doing.
4. Added ability to process data in chunks.
5. Added print statements that signal the progress of a given step.
6. Reformatted code to be more in line with PEP standards.
7. Added ability to save and load from track bank files.
8. Default source file (file where tracks come from) was changed to a file with
    more pronounced track curvature.
9. Events generated now have a uniform distribution of track sizes.
    So, for example, if you wanted 300 events with a max track size of 10,
    then there would be 30 events with just one track, 30 events with two
    tracks, ..., and 30 events with 10 tracks. Previously, track sizes were
    generated randomly using the poisson distribution and a specified *mu*
    average value.
"""

import os
import glob
import pickle
import h5py
import time
import numpy as np
import pandas as pd
import itertools
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


class DatasetDealer:
    """ A DatasetDealer. """
    def __init__(self,
                 filename: str="../datasets/public_train_100MeV.csv",
                 label: str="",
                 bankfile: str="",
                 outputfile: str="../datasets/generated_uniform.csv")\
            -> None:
        """ Initialize a DatasetDealer. """
        self.TRACK_BANK_NAME  = "../datasets/track_bank.csv"
        self.OUTPUT_FILE_NAME = outputfile

        self.filename = filename
        self.label    = label
        self.bankfile = bankfile
        self.tf       = None

        self.all_phi  = [
            9802,  21363,  38956,
            53533, 68110,  50894,
            70624, 95756, 125664
        ]

    def create_bank(self,
                    parse: int=2500,
                    n_chunks: int=100)\
            -> None:
        """ Create a bank. """
        if type(self.tf) == pd.core.frame.DataFrame:
            return  # Looks like we have already initialized *self.tf*.

        if self.bankfile:
            self.tf = pd.read_csv(self.bankfile)
            print("Loaded bank from file.")
            return

        print("Retrieving a track bank of size {}.".format(parse))

        df        = pd.read_csv(self.filename)
        max_count = 0
        index     = 0
        chunks    = [pd.DataFrame(columns=(
            'track_id',
            'layer',
            'iphi',
            'x',
            'y'))
            for _ in range(n_chunks)]

        uniques = pd.unique(df["event_id"])
        np.random.shuffle(uniques)
        uniques = uniques[:parse]
        for i, event_id in enumerate(uniques):
            sub_data = df.loc[df['event_id'] == event_id]
            augment  = defaultdict(list)
            rows     = zip(
                    sub_data['x'].values,
                    sub_data['y'].values,
                    sub_data['layer'].values,
                    sub_data['iphi'].values,
                    sub_data['cluster_id'].values
            )
            for (x, y, layer, iphi, aclass) in rows:
                augment[max_count + aclass].append([
                    int(max_count + aclass),
                    int(layer),
                    int(iphi),
                    x,
                    y
                ])
            max_count += max(sub_data['cluster_id'].values) + 1
            for trackid in augment:
                for point in sorted(augment[trackid], key=lambda k: k[1]):
                    chunks[i % n_chunks].loc[index] = point
                    index += 1
            print("\rCount: {0} / {1}".format(i, parse), end="")
        print("")
        self.tf = pd.concat(chunks)
        self.tf.to_csv(self.TRACK_BANK_NAME, index=False)

    def create_dataset(self,
                       n: int=100,
                       max_track_per_event=25,
                       n_chunks: int=10)\
            -> None:
        filename = self.OUTPUT_FILE_NAME
        print("Creating data of size {0} into '{1}'.".format(n, filename))
        frame = self.generate(n, max_track_per_event, n_chunks)
        frame.to_csv(filename, index=False)
        s = set(frame["event_id"])
        print("Created range of events {0} to {1}".format(min(s), max(s)))

    def generate(self, n: int, max_track_per_event: int, n_chunks: int):
        self.create_bank()
        print("Generating {} events.".format(n))
        track_index = set(self.tf['track_id'].values)
        n_tracks = np.array([(i % max_track_per_event) + 1 for i in range(n)])
        np.random.shuffle(n_tracks)
        chunks = [pd.DataFrame(columns=(
            'event_id',
            'cluster_id',
            'layer',
            'iphi',
            'x',
            'y'))
            for _ in range(n_chunks)
        ]
        master_index = 0
        for ievent, n_track in enumerate(n_tracks):
            phis = np.random.random(size=(n_track,)) * np.pi * 2

            # pick at random n_track in the bank
            for i_track, track_id in enumerate(
                    np.random.choice(list(track_index), n_track)):
                # the angle by which the track is going to be rotated
                phi = phis[i_track]

                # [0:'track_id', 1:'layer', 2:'iphi',3:'x',4:'y']
                track_data = self.tf.loc[self.tf['track_id'] == track_id].values

                # modify iphi accordingly
                for i, layer in enumerate(track_data[:, 1]):
                    track_data[i, 2] = (int(track_data[i, 2]
                                            + self.all_phi[int(layer)]
                                            * phi
                                            / (np.pi * 2))
                                        % self.all_phi[int(layer)])
                x = track_data[:, 3]
                y = track_data[:, 4]

                # rotate the initial hit coordinates in x,y (N.B. this is not
                # used eventually in the model. so we can skip it for speed-up
                c, s  = np.cos(phi), np.sin(phi)
                r     = np.matrix([[c, -s], [s, c]])
                coord = np.matrix([x, y])
                coord = np.dot(r, coord)

                # rotated track
                track_data[:, 3] = np.ravel(coord[0, ...])
                track_data[:, 4] = np.ravel(coord[1, ...])
                for entry in range(track_data.shape[0]):
                    l = np.zeros((6,))
                    l[:2] = [ievent, i_track]
                    l[2:] = track_data[entry, 1:]
                    chunks[master_index % n_chunks].loc[master_index] = l
                    master_index += 1
            print("\rCount: {0} / {1}".format(ievent + 1, n), end="")
        print("")
        # cast the types of the table
        data = pd.concat(chunks)
        data['event_id']   = data['event_id'].astype(int)
        data['layer']      = data['layer'].astype(int)
        data['cluster_id'] = data['cluster_id'].astype(int)
        data['iphi']       = data['iphi'].astype(int)
        return data

if __name__ == '__main__':
    print("Starting program.")

    # First, create a DatasetDealer that loads tracks from the file, *filename*.
    # If a *bankfile* name is specified, the track bank will be the tracks
    # stored within that file, *bankfile*.
    d = DatasetDealer(
            bankfile="../datasets/track_bank.csv",
            outputfile="../datasets/tester.csv"
    )

    # If a *bankfile* was not specified, we are going to have to create our own
    # *bankfile*.
    if not d.bankfile:
        d.create_bank(parse=3600, n_chunks=720)

    # Finally, create a data set and save it.
    d.create_dataset(n=100, max_track_per_event=5, n_chunks=10)
    print("Ending program.")
