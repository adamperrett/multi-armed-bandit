import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import time
import pylab
import numpy as np
from threading import Condition
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pyNN.random import RandomDistribution as rand
#import spynnaker8.spynakker_plotting as splot
import csv
import pandas

#define the parameters


#initialise population or possibly read in from text file

#different neuron types/characteristics
    #exite inhib
    #types/P() of connectors to other neuron markers
    #pop size?
    #neuron parameter ranges
    #reactivity to chemical gradients

#chemical gradients
    #decay constant and shape (in each dimention or uniform)
    #attractive/repulsive to certain neuron markers and degree?
    #initial concentration

#3D map orientation - maybe seperate for neurons and chemical
    #position on input and output neural populations
    #postion in the discrete 3D field of neurons
    #postion in the discrete 3D field of checmical secreaters
    #number of seperate chemical secreters
    #number of neural populations
    #size of the field/map
    #no cross breeding between maps of different size or number of 'population'


#test population (all combos of 3 evo properties, or pos not depends on construction)
    #many combinations of ball and beam starting point
    #roll together if time is an issue
    #random initial conditions?
    #random test ordering to build robustness
    #average distance^2 from the centre assuming non random tests

#evolve on property keeping the others fixed
    #select the best and evolve against it
    #or keep a few of the best and evaluate them in combination

#