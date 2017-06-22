import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    
    def load_data(self, num_events, tracks_per_event, track_size, noise_per_event):
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
        labels = ["phi", "r", "z"]
        groups = self.dataframe[["id", "r", "phi", "z"]].groupby("id")
        goods  = groups.filter(lambda track: len(track) == track_size)
        bads   = groups.filter(lambda track: len(track) != track_size)
        
        # Populate input and output with data.
        goods_group = [g[1] for g in list(goods.groupby("id"))]
        self.input  = np.zeros((num_events, hits_per_event, len(labels)))
        self.output = np.zeros((num_events, hits_per_event, tracks_per_event))
        for n in range(num_events):
            # Retrieve a sample of tracks.
            tracks = random.sample(goods_group, tracks_per_event)
            
            # Make a mapping from track ID to index within a matrix.
            T2I = self.__get_matrix_map__(tracks) # Track to Index
            
            # Make some noise hits to add.
            noise  = bads.sample(noise_per_event)
            hits   = pd.concat(tracks + [noise]).sort_values(labels)
            
            self.__populate_input__(hits, labels, n)
            self.__populate_output__(hits, T2I, n)
    # END FUNCTION load_data
    
    def __populate_input__(self, hits, labels, event_index):
        self.input[event_index, :] = hits[labels].values
    # END FUNCTION __populate_input__
    
    def __populate_output__(self, hits, mapping, event_index):
        for t, track_ID in enumerate(hits["id"]):
            index = mapping.get(track_ID)
            if index is not None:
                self.output[event_index, t, index] = 1
    # END FUNCTION __populate_output__
        
    def __get_matrix_map__(self, tracks):
        L = pd.concat([T.sort_values(["r"]).head(1) for T in tracks])
        L.sort_values(["phi", "z"], inplace=True)
        T2I = dict()
        for idx, hit in enumerate(L["id"]):
            T2I[hit] = idx
        return T2I
    # END FUNCTION __get_matrix_map__
# END CLASS LinearTracker

filename  = ('../Data Sets/linear_data_5k.csv')
dataframe = pd.read_csv(filename)
tracker   = LinearTracker(dataframe)
np.random.seed(7)
tracker.load_data(num_events=2, tracks_per_event=5, track_size=4, noise_per_event=5)
print(tracker.output)