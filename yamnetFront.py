# -*- coding: utf-8 -*-
"""
Yamnet visualization via Tkinter for Signal Processing Innovation Project
at Tampere University, spring 2020.
"""
#Import matplotlib and relevant child libraries
from __future__ import division, print_function
import matplotlib

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style
import matplotlib.pyplot as plt

#Import tkinter and its style pack ttk for more pleasant buttons if needed
import tkinter as tk
#from tkinter import ttk

#Import numpy for miscellanious number/data crunching
import numpy as np
from recorder import Recorder


import soundfile as sf
import params
import yamnet as yamnet_model
import resampy



import time

"""
Tkinter base class. The functionality could be done as a wrapper, but in order
to combine example codes, I've strayed from the best practice
"""
class tkyamnet(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        #Constructor, builds the tkinter app and used frames
        
        #Run the base class init        
        tk.Tk.__init__(self, *args, **kwargs)
        
        """TODO Initialize Yamnet"""
        self.yamnet = yamnet_model.yamnet_frames_model(params)
        self.yamnet.load_weights('yamnet.h5')
        
        #Prepare the visualization graph. Tight layout for fitting better
        self.figure, self.axs = plt.subplots(10, figsize=(10,10))
        plt.tight_layout()
        
        
        #Prepare colors. Colors are xkcd-colors in random order    
        with open('colors.txt', 'r') as colorfile:
            self.colors = colorfile.readlines()
               
        #Strip newlines for more efficient use
        for i in range(len(self.colors)):
            self.colors[i] = self.colors[i][:-1]
            
        #Prepare Yamnet class names    
        with open('classes.txt', 'r') as classesfile:
            self.classes = classesfile.readlines()
            
        #Strip newlines for more efficient use
        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i][:-1]
        
        #Base frame to build the used frames from
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        
        #Dict of used frames
        self.frames = {}
        
        #Build each used frame, initialize a grid for them
        for F in (GraphPage,):
        
            frame = F(container, self)          
            self.frames[F] = frame          
            frame.grid(row = 0, column = 0, sticky="nsew")
            
        
        #Bring StartPage on top for user
        self.show_frame(GraphPage)
        
        #Declare class variables used in animation   
        self.xList = np.linspace(-30, -1, 30)
        
        #Prepare the yamnet-format results, 521 classes for 30 seconds
        self.data = np.zeros((521,30))
        #Prepare the weights used to rank classification results
        self.scores = np.zeros(521)
        
        #Start audio recording
        self.rec = Recorder(channels=1)
        self.recfile = self.rec.open('sample.wav','wb')
        self.recfile.start_recording()
        #After a second, start animating 
        self.after(1000, self.animate)
        
    def show_frame(self, cont):
        
        #Puts the frame of the class passed as cont on top
        
        frame = self.frames[cont]
        frame.tkraise()
        
    def close_windows(self):
        
        #Closes the Tkinter and any pending events
        
        self.quit()
        self.destroy()
        self.recfile.close()
        
    def animate(self):
        
        time_of_start = time.time()
        #Runs Yamnet visualization, queued as an event every ~second
        
        self.recfile.stop_recording()
        self.recfile.close()
        
        #Reading sample.wav file which is a mono signal (used in Yamnet by default) into desired format
        wav_data, sr = sf.read('sample.wav', dtype=np.int16)
        
        self.recfile = self.rec.open('sample.wav','wb')
        self.recfile.start_recording()
        
        #Queue another iteration a second from now
        self.after(1000, self.animate)
        
        
        new_samples = self.classification(wav_data)
        
        #The ranking is decided by IIR-filtered samples
        self.scores = self.scores * 0.9 + new_samples
        
        #Expand new sample dimensions for concatenation
        new_samples = np.expand_dims(new_samples, axis=1)
        
        #Concatenate the new samples to right, remove leftmost column
        self.data = np.concatenate((self.data, new_samples), axis=1)
        self.data = np.delete(self.data, 0, 1)
        
        #Find the indexes of top 10 scores
        indexes = self.findtopX(self.scores, 10)
           
        #Plot each subplot with corresponding class data
        for i in range(10):
            
            #Select subplot
            a = self.axs[i]
            
            #Remove old plot
            a.clear()
            
            #Plot class scores of last 30 seconds
            a.plot(self.xList, self.data[indexes[i], :], 'xkcd:'+self.colors[indexes[i]])
            #Set constant axis so that confidence close to 1 is plotted fully
            a.set_ylim((0, 1.1))
            
            
            #Set the label next to subplot as class name
            self.frames[GraphPage].labels[i]['text'] = self.classes[indexes[i]]
             
        
        #Draw canvas to show updated graph
        self.frames[GraphPage].canvas.draw()
        
        # print(time.time() - time_of_start)
        
        
    def findtopX(self, values, X):
        
        #Find the indexes of largest X values from an 1D iterable list
        
        #Variables that keep track of the topX values and indexes
        topX = [0] * X
        indexes = [0] * X
        
        #Iterate trough the whole list
        for index in range(len(values)):
            """
            Value is compared to the lowest uncompared value each iteration.
            If candidate value is bigger, swap the ranking of comparables
            and then compare to the next lowest value.
            If candidate is smaller, move to next candidate.
            """
            
            if values[index] > topX[X-1]:
                topX[X-1] = values[index]
                indexes[X-1] = index
            else:
                #Skip to next value
                continue
            
            for i in range(X-2, -1, -1):       
                
                if values[index] > topX[i]:
                    
                    topX[i+1] = topX[i]
                    indexes[i+1] = indexes[i]
                    
                    topX[i] = values[index]
                    indexes[i] = index
                else:
                    #Skip to next value
                    break
        
        return indexes
    
    def classification(self,wav_data):
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        sr = 44100
        
        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != params.SAMPLE_RATE:
            waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)
        # Predict YAMNet classes.
        # Second output is log-mel-spectrogram array (used for visualizations).
        # (steps=1 is a work around for Keras batching limitations.)
        scores, _ = self.yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
        # Scores is a matrix of (time_frames, num_classes) classifier scores.
        # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores, axis=0)
        # Report the highest-scoring classes and their scores.
        #sound_events = np.argsort(prediction)[::-1]
        return prediction
            
                 
'''
#A start page for navigating, if necessary
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, width = 400, height = 400)
        
            
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack()
        
        button1 = ttk.Button(self, text="To Graph", command=lambda:controller.show_frame(GraphPage))
        button1.pack()
        
        button2 = ttk.Button(self, text="Exit", command=lambda:controller.close_windows())
        button2.pack()
        
        record_Button = ttk.Button(self, text='Start', command=self.record_sample() )
        record_Button.pack()
        
        stop_Button = ttk.Button(self, text='Start', command=self.record_sample() )
        stop_Button.pack()
'''
       
        
class GraphPage(tk.Frame):
    def __init__(self, parent, controller):
        #Run parent initialization
        tk.Frame.__init__(self, parent)
        
        self.labels = []
        
        #Configure grid layout
        for i in range(11):
            self.rowconfigure(i, weight = 1)
            
        for i in range(7):
            self.columnconfigure(i, weight = 1)
        
        #Labels on top of graph
        self.titlelabel = tk.Label(self, text="Yamnet audio classification", font=("Verdana", 12))
        self.titlelabel.grid(row = 0, column = 0, columnspan=3, sticky = 'nsew')
        
        self.scorelabel = tk.Label(self, text="Top 10 scores from Yamnet", font=("Verdana", 12))
        self.scorelabel.grid(row = 0, column = 5, columnspan=2, sticky = 'nsew')

        """
        button1 = ttk.Button(self, text="To Home", command=lambda:controller.show_frame(StartPage))
        button1.grid(row = 0, column = 1, sticky = 'nsew')
        """
        
        #Create canvas for displaying the figure
        self.canvas = FigureCanvasTkAgg(controller.figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=10, columnspan=5, sticky = "nsew")
        
        #Labels for class names. These could've been procedurally generated, but meh
        
        label1 = tk.Label(self, text = "Number 1", font= ("Verdana", 12))
        label1.grid(row = 1, column = 5, columnspan=2, sticky = "nsew")
        self.labels.append(label1)
        
        label2 = tk.Label(self, text = "Number 2", font= ("Verdana", 12))
        label2.grid(row = 2, column = 5, columnspan=2, sticky = "nsew")
        self.labels.append(label2)
        
        label3 = tk.Label(self, text = "Number 3", font= ("Verdana", 12))
        label3.grid(row = 3, column = 5, columnspan=2, sticky = "nsew")
        self.labels.append(label3)
        
        label4 = tk.Label(self, text = "Number 4", font= ("Verdana", 12))
        label4.grid(row = 4, column = 5, columnspan=2, sticky = "nsew")
        self.labels.append(label4)
        
        label5 = tk.Label(self, text = "Number 5", font= ("Verdana", 12))
        label5.grid(row = 5, column = 5, columnspan=2, sticky = "nsew")       
        self.labels.append(label5)
        
        label6 = tk.Label(self, text = "Number 6", font= ("Verdana", 12))
        label6.grid(row = 6, column = 5, columnspan=2, sticky = "nsew")       
        self.labels.append(label6)
        
        label7 = tk.Label(self, text = "Number 7", font= ("Verdana", 12))
        label7.grid(row = 7, column = 5, columnspan=2, sticky = "nsew")       
        self.labels.append(label7)
        
        label8 = tk.Label(self, text = "Number 8", font= ("Verdana", 12))
        label8.grid(row = 8, column = 5, columnspan=2, sticky = "nsew")       
        self.labels.append(label8)
        
        label9 = tk.Label(self, text = "Number 9", font= ("Verdana", 12))
        label9.grid(row = 9, column = 5, columnspan=2, sticky = "nsew")       
        self.labels.append(label9)
        
        label10 = tk.Label(self, text = "Number 10", font= ("Verdana", 12))
        label10.grid(row = 10, column = 5, columnspan=2, sticky = "nsew")       
        self.labels.append(label10)
        

if __name__ == "__main__":   

    #Use Tkinter compatible matplotlib version and style
    matplotlib.use("TkAgg")
    style.use("ggplot")
    
    #Do not visualize figures via IPython, only via Tkinter
    plt.ioff()
    
    #Construct TKinter app                
    app = tkyamnet()

    #Use custom deconstructor
    app.protocol('WM_DELETE_WINDOW', app.close_windows)

    #Start looping in app
    app.mainloop()

    #Close the figures in memory as well, otherwise these pile up
    plt.close('all')
