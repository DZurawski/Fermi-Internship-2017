import collections
import pandas as pd
import numpy as np

class ProbabilityMatrix():
    """ A wrapper class to a probability matrix provided by a LinearTracker. """
    def __init__(self, matrix, track_IDs, hit_IDs):
        """ Initialize a ProbabilityMatrix.
            @param matrix - 2-D numpy array: A probability matrix such that
                each row represents a track, each column represents a hit and
                an element represents the probability that a hit is a member of
                track.
            @param track_IDs - list of ints: The list of all track IDs.
            @param hit_IDs - list of ints: The list of all hit IDs.
            @return Nothing
        """
        self.matrix  = matrix # The underlying matrix containing probabilities.
        self.__T2I__ = collections.OrderedDict() # Maps track ID to an index within self.matrix.
        self.__H2I__ = collections.OrderedDict() # Maps hit ID to an index within self.matrix.
        
        for i, track_ID in enumerate(track_IDs):
            self.__T2I__[i] = track_ID
        
        for i, hit_ID in enumerate(hit_IDs):
            self.__H2I__[i] = hit_ID
    
    def probability(self, track_ID, hit_ID):
        """ Return the probability that the particle with 'hit_ID' is in track with 'track_ID'.
            @param track_ID - int: A track ID
            @param hit_ID - int: A hit ID
            @return float 
                A probability float within [0, 1].
        """
        track_idx = self.__T2I__[track_ID]
        hit_idx   = self.__H2I__[hit_ID]
        return self.matrix[track_idx, hit_idx]
    
    def probabilities(self, hit_ID):
        """ Return a list of track ID and a vector of probabilites.
            @param hit_ID - int: The ID of a particle
            @return ([int], 1-D numpy array)
                A list of track ID and a vector of probabilities such that
                the index of a track ID within the list corresponds to the index
                of the probability within the probability vector.
        """
        hit_idx = self.__H2I__[hit_ID]
        return (self.__T2I__.keys(), self.matrix[:, hit_idx])
# END CLASS ProbabilityMatrix

class LinearTracker():
    """ An object that classifies particles to tracks after an event. """
    def __init__(self):
        """ Initialize a LinearTracker.
            @param filename - string: The name of a file to load data from.
            @param model - keras model - A network model that the tracker will use to
                classify particles.
            @return Nothing
        """
        self.model       = None  # keras model
        self.dataframe   = None  # pandas DataFrame
        self.test_input  = None  # list of events. Event is list of tracks. Track is list of position vectors.
        self.test_output = None  # ???
    
    def load_dataframe(self, dataframe, tracks_per_event=5):
        """ Load a data frame to this tracker.
            @param dataframe - pandas DataFrame - the data frame to load from.
            @param tracks_per_event - int - The number of tracks per event.
            @return Nothing
        """
        self.dataframe = dataframe
        groups = self.dataframe.groupby("id") # Hits grouped by track ID.
        tracks = [group[1] for group in groups] # Track data - list of pd.DataFrame
        TPE, L = tracks_per_event, dataframe.size # Renames for brevity.
        events = [tracks[i:i+TPE] for i in range(0, L, TPE)] # Sliced track data
        self.test_input  = [[track[["z", "phi", "r"]].values for track in event] for event in events]
        
        
        
        
        for i, event in enumerate(self.test_input):
            print("Event {}".format(i))
            for j, track in enumerate(event):
                print("Track {}".format(j))
                print(track)
        
        #self.__extract_test_input__ (tracks, tracks_per_event)
        #self.__extract_test_output__(tracks, tracks_per_event)
    
    def __extract_test_input__(self, tracks, tracks_per_event):
        positions = [track[["z", "phi", "r"]].values for track in tracks]
        TPE, L    = tracks_per_event, len(positions) # For brevity
        self.test_input = [positions[i:i+TPE] for i in range(0, L, TPE)]
         
        for i, event in enumerate(self.test_input):
            print("Event {}".format(i))
            for j, track in enumerate(event):
                print("Track {}".format(j))
                print(track)
    
    def __extract_test_output__(self, tracks, tracks_per_event):
        pass
# END CLASS LinearTracker
       
def main():
    np.random.seed(7)
    filename  = ('file_o_stuff3.csv')
    tracker   = LinearTracker()
    dataframe = pd.read_csv(filename)
    tracker.load_dataframe(dataframe)
# END FUNCTION main


print("Starting the program.")
main()
print("Ending the program.")