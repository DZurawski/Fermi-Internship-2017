import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax

pd.options.display.float_format = '{:,.2f}'.format

class LinearTracker():
    """ An object that classifies particles to tracks after an event. """    
    def __init__(self, dataframe, model=None):
        """ Initialize a LinearTracker.
            @param dataframe - pd.DataFrame - used to pick tracks from.
                The headers should contain: ("id", "act_z", "r", "phi").
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
        labels = ["phi", "r", "act_z"]
        groups = self.dataframe[["id", "r", "phi", "act_z"]].groupby("id")
        goods  = groups.filter(lambda track: len(track) == track_size)
        bads   = groups.filter(lambda track: len(track) != track_size)
        
        # Populate input and output with data.
        goods_group = [g[1] for g in list(goods.groupby("id"))]
        self.input  = np.zeros((num_events, hits_per_event, len(labels)))
        self.output = np.zeros((num_events, hits_per_event, tracks_per_event + 1))
        for n in range(num_events):
            # Retrieve a sample of tracks.
            tracks = random.sample(goods_group, tracks_per_event)
            
            # Make a mapping from track ID to index within a matrix.
            T2I = self.__get_matrix_map__(tracks) # Track to Index
            
            # Make some noise hits to add.
            noise  = bads.sample(noise_per_event)
            hits   = pd.concat(tracks + [noise]).sort_values(labels)
            
            self.__populate_input__(hits, labels, n)
            self.__populate_output__(hits, T2I, n, tracks_per_event)
    # END FUNCTION load_data
    
    def plot(self, event_index=None, in_data=None, out_data=None):
        """ Display a 3D plot of the event with event index 'eventID'.
        """
        if event_index is not None:
            in_data  = self.input[event_index]
            out_data = self.output[event_index]
        elif in_data is None or out_data is None:
            print("Please provide an event_index argument.")
            return
        
        # Convert the input (phi, r, z) to cartesian (x, y, z)
        conv = lambda PRZ : (np.cos(PRZ[0])*PRZ[1],np.sin(PRZ[0])*PRZ[1],PRZ[2])
        XYZ  = np.array([conv(hit) for hit in in_data])
        
        # Get the colors.
        cmap = plt.cm.get_cmap('hsv', out_data.shape[1])
        tracks, colors = self.__trackify__(XYZ, out_data, cmap)

        # Create the plot.
        plot = plt.figure().add_subplot(111, projection='3d')        
        for i, track in enumerate(tracks[:-1]):
            plot.plot(xs=[t[0] for t in track],
                      ys=[t[1] for t in track],
                      zs=[t[2] for t in track],
                      c=colors[i], linestyle='-', marker='o')
        plot.scatter(xs=[t[0] for t in tracks[-1]],
                     ys=[t[1] for t in tracks[-1]],
                     zs=[t[2] for t in tracks[-1]],
                     c=colors[-1])
        plt.show(plot)
    # END FUNCTION create_plot
    
    def __trackify__(self, in_data, out_data, cmap):
        """
            @return a pair such that:
                pair[0] - list of tracks, where a track is a list of hits.
                pair[1] - list of colors with index corresponding to how
                    to color the track at that index.
        """
        indices = [np.argmax(out_data[i]) for i in range(in_data.shape[0])]
        tracks  = [[] for _ in range(out_data.shape[1])]
        for i, hit in enumerate(in_data):
            tracks[indices[i]].append(hit)
        tracks = np.array(tracks)
        colors = np.array([cmap(i) for i in range(len(indices))])
        return (tracks, colors)
    # END FUNCTION __trackify__
    
    def show_losses(self, histories):
        """ Graph the accuracy and loss of a model's histories.
            Code from HEPTrks keras tutorial file. in DSHEP folder.
            @param histories - list of pairs (string, history from model) 
            @return Nothing
        """
        plt.figure(figsize=(10,10))
        plt.ylim(bottom=0)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Error by Epoch')
        colors=[]
        do_acc=False
        for label,loss in histories:
            color = tuple([0.1, 0.1, 0.1])
            colors.append(color)
            l = label
            vl= label+" validation"
            if 'acc' in loss.history:
                l+=' (acc %2.4f)'% (loss.history['acc'][-1])
                do_acc = True
            if 'val_acc' in loss.history:
                vl+=' (acc %2.4f)'% (loss.history['val_acc'][-1])
                do_acc = True
            plt.plot(loss.history['loss'], label=l, color=color)
            if 'val_loss' in loss.history:
                plt.plot(loss.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)
        plt.legend()
        plt.yscale('log')
        plt.show()
        if not do_acc: return
        plt.figure(figsize=(10,10))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        for i,(label,loss) in enumerate(histories):
            color = tuple([0.0, 0.0, 1.0])
            if 'acc' in loss.history:
                plt.plot(loss.history['acc'], lw=2, label=label+" accuracy", color=color)
            if 'val_acc' in loss.history:
                plt.plot(loss.history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
        plt.legend(loc='lower right')
        plt.show()
    # END FUNCTION show_losses
       
    def __populate_input__(self, hits, labels, event_index):
        """ Populate the input at event with index 'event_index'.
            @param hits - pd.DataFrame
                The pd.DataFrame of hits to set this event to.
            @param labels - The categories we want from the hits pd.DataFrame.
            @param event_index - int
                Index of the event to set.
            @return Nothing
        """
        self.input[event_index, :] = hits[labels].values
    # END FUNCTION __populate_input__
    
    def __populate_output__(self, hits, mapping, event_index, tracks_per_event):
        """ Populate the output at event with index 'event_index'.
            @param hits - pd.DataFrame
                The pd.DataFrame of hits to set this event to.
            @param mapping - dictionary object (int -> int)
                A dictionary object that maps track ID to matrix index.
            @param event_index - int
                Index of the event to set.
            @param tracks_per_event - int
                The number of tracks per event.
                The last column (index: tracks_per_event) is the noise column.
            @return Nothing
        """
        noise_index = tracks_per_event
        for t, track_ID in enumerate(hits["id"]):
            index = mapping.get(track_ID)
            if index is not None:
                self.output[event_index, t, index] = 1
            else:
                self.output[event_index, t, noise_index] = 1
    # END FUNCTION __populate_output__
        
    def __get_matrix_map__(self, tracks):
        """ Get a dictionary that maps track ID to matrix index.
            @param tracks - list of pd.DataFrames
                Each pd.DataFrame consists of its hits.
            @return dictionary object (int -> int)
                A mapping from track ID to matrix index.
        """
        L = pd.concat([T.sort_values(["r"]).head(1) for T in tracks])
        L.sort_values(["phi", "act_z"], inplace=True)
        return dict((hit, idx) for idx, hit in enumerate(L["id"]))
    # END FUNCTION __get_matrix_map__
# END CLASS LinearTracker

filename  = ('../Data Sets/corrected5k_data.csv')
dataframe = pd.read_csv(filename)
tracker   = LinearTracker(dataframe)

np.random.seed(7)
tracker.load_data(num_events=2, tracks_per_event=5, track_size=4, noise_per_event=5)
tracker.plot(event_index=0)
# print("Ding! All done.")
# winsound.Beep(2200, 1000)
# winsound.Beep(1800, 1000)
# winsound.Beep(1000, 1000)
# 
# from IPython.display import display,HTML
# 
# def multi_column_df_display(list_dfs, cols=2):
#     """ Code by David Medenjak responding to StackOverflow question found here:
#         https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
#         Displays a list of dataframes in IPython as a table with cols number of columns.
#     """
#     html_table = "<table style='width:100%; border:0px'>{content}</table>"
#     html_row = "<tr style='border:0px'>{content}</tr>"
#     html_cell = "<td style='width:{width}%;vertical-align:top;border:0px'>{{content}}</td>"
#     html_cell = html_cell.format(width=100/cols)
# 
#     cells = [ html_cell.format(content=df.to_html()) for df in list_dfs ]
#     cells += (cols - (len(list_dfs)%cols)) * [html_cell.format(content="")] # pad
#     rows = [ html_row.format(content="".join(cells[i:i+cols])) for i in range(0,len(cells),cols)]
#     display(HTML(html_table.format(content="".join(rows))))
# # END FUNCTION multi_column_df_display
# 
# input_cols  = ["phi", "r", "z"]
# output_cols = ["T{}".format(i) for i in range(tracker.output.shape[2] - 1)] + ["N"]
# show_max    = 2
# 
# if show_max is not None and show_max > 0 and show_max < len(tracker.input):
#     print("Displaying the first {} inputs and outputs.".format(show_max))
#     input_frames  = [pd.DataFrame(data=tracker.input[i], columns=input_cols) for i in range(show_max)]
#     output_frames = [pd.DataFrame(data=tracker.output[i].astype(int), columns=output_cols) for i in range(show_max)]
# else:
#     print("Displaying all of input and output.")
#     input_frames  = [pd.DataFrame(data=matrix, columns=input_cols)  for matrix in tracker.input]
#     output_frames = [pd.DataFrame(data=matrix.astype(int), columns=output_cols) for matrix in tracker.output]
#     
# df_list  = []
# for i in range(len(input_frames)):    
#     df_list.append(input_frames[i])
#     df_list.append(output_frames[i])
# 
# print("Input shape:  {}".format(tracker.input.shape))
# print("Output shape: {}".format(tracker.output.shape))
# multi_column_df_display(df_list)
# 
# from keras.layers import Dense, LSTM, Dropout
# from keras.models import Sequential
# 
# input_shape = tracker.input[0].shape # Shape of an event.
# output_shape = len(tracker.output[0][0]) # Number of tracks per event
# 
# batch_size = 32
# epochs     = 256
# valsplit   = 0.25
# opt        = 'rmsprop' # optimizer
# tracker.model = Sequential()
# tracker.model.add(LSTM(32, return_sequences=True, input_shape=input_shape, dropout=.2, recurrent_dropout=.2))
# tracker.model.add(Dense(output_shape, activation='softmax'))
# 
# tracker.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# #tracker.model.summary()
# 
# modelpath = 'simple.h5'
# hist = tracker.model.fit(tracker.input, tracker.output, epochs=epochs, batch_size=batch_size,
#                          verbose=0, validation_split=valsplit,
#                          callbacks=[keras.callbacks.ModelCheckpoint(filepath=modelpath, verbose=0)])
# print("Ding! All done.")
# winsound.Beep(1000, 1000)
# winsound.Beep(1800, 1000)
# winsound.Beep(2200, 1000)
# 
# score, acc = tracker.model.evaluate(tracker.input, tracker.output, batch_size=batch_size)
# print("\nTest Score:    {}".format(score))
# print("Test Accuracy: {}".format(acc))
# show_losses([("Categorical Cross Entropy", hist)])
# 
# predictions = tracker.model.predict(tracker.input[:len(input_frames)], batch_size=batch_size)
# df = []
# for i in range(len(input_frames)):
#     df.append(input_frames[i])
#     df.append(pd.DataFrame(data=predictions[i], columns=output_cols))
# multi_column_df_display(df)