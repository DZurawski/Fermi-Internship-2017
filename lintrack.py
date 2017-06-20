""" lintrack.py
"""

import random
import pandas as pd
import numpy as np

class LinearTracker():
    """ An object that classifies particles to tracks after an event. """    
    def __init__(self, dataframe, model=None):
        """ Initialize a LinearTracker.
            @param dataframe - pd.DataFrame - used to pick tracks from.
                The headers should contain: ("id", "z", "r", "phi").
            @param model - keras model - A network model that the tracker will
                use to classify particles.
            @return Nothing
        """
        self.model     = model     # keras model to figure out tracks.
        self.dataframe = dataframe # pandas.DataFrame for picking tracks.
        self.input     = None      # input to train model on.
        self.output    = None      # output to train model on.
    # END function __init__
    
    def load_data(self, num_events=1,
                  tracks_per_event=3, track_size=4, noise_per_event=3):
        """ Load input and output data from this object's dataframe.
            @param num_events - int - The number of events to generate.
            @param tracks_per_event - int - The number of tracks per event.
            @param track_size - int - The number of hits per track.
            @param noise_per_event - int - The number of hits with no track.
            @return Nothing
                However, self.input and self.output become numpy arrays.
                self.input is collection of hits of shape:
                    (num_events, hits_per_event, 3)
                self.output is list of probability matrices of shape:
                    (num_events, hits_per_event, tracks_per_event)
        """
        hits_per_event = (track_size * tracks_per_event) + noise_per_event
        data   = self.dataframe[["id", "r", "phi", "z"]].drop_duplicates()
        groups = data.groupby("id")
        valids = groups.filter(lambda track: len(track) == track_size)
        bads   = groups.filter(lambda track: len(track) != track_size)
        labels = ["phi", "r", "z"]
        
        # Populate input and output with data.
        self.input  = np.zeros((num_events, hits_per_event, len(labels)))
        self.output = np.zeros((num_events, hits_per_event, tracks_per_event))
        for n in range(num_events):
            # Retrieve the hits within this event.
            sample = random.sample(list(valids.groupby("id")), tracks_per_event)
            tracks = [track[1] for track in sample] # Make it not a tuple.
            noise  = bads.sample(noise_per_event)
            hits   = pd.concat(tracks + [noise])
            hits.sort_values(labels, inplace=True)
            
            # Populate this event's inputs.
            self.input[n, :] = hits[labels].values
            
            # Define a mapping from track ID to probability matrix column.
            T2I = dict()
            for t, track_ID in enumerate([s[0] for s in sample]):
                T2I[track_ID] = t
            
            # Populate this event's outputs.
            for t, track_ID in enumerate(hits["id"]):
                index = T2I.get(track_ID)
                if index is not None:
                    self.output[n, t, index] = 1
    # END FUNCTION load_data
# END CLASS LinearTracker
       
def main():
    np.random.seed(7)
    filename  = ('file_o_stuff3.csv')
    dataframe = pd.read_csv(filename)
    tracker   = LinearTracker(dataframe)
    tracker.load_data()
    
    print(tracker.input)
    print("- - - - -- - - - - - -- - - - - - -")
    print(tracker.output)
# END FUNCTION main

print("Starting the program.")
main()
print("Ending the program.")