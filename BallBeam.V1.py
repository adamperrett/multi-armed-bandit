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

#define the network parameters
neuron_pop_size = 10
chem_pop_size = 10
map_pop_size = 10
marker_length = 7

#define experimental paramters
beam_length = 2 #centred half way
g = 9.81
mass = 0.01
radius = 0.01
moi_ball = 0.0000006
moi_beam = 0.02

#initialise population or possibly read in from text file

#different neuron types/characteristics
    #exite inhib
    #types/P() of connectors to other neuron markers
    #pop size?
    #neuron parameter ranges
    #reactivity to chemical gradients

#excititory_prob = 0-1
#connection_prob = 0-1 #per other neuron type?
    #dendrite receptive field (also guided by chemicals
        # http://neuralensemble.org/docs/PyNN/0.7/standardmodels.html
    # v_rest -65mV
    # cm 1nF
    # tau_m 20ms
    # tau_refract 0ms
    # tau_syn_E 5ms
    # tau_syn_I 5ms
    # i_offset 0nA
    # v_reset -65mV
    # v_thresh -50mV
#weight mean = 0.015 (max 0.03?)
#weight stdev = 0.05
    #plasticty
#delay mean = 30 (max 60?)
#delay stdev = 10
#chem level to stop
#noise of chem detection
    #neuron_density = 0-100
    # reactivity = bit string as a number representing which neurons it is attracted to
#chemical_marker = 2^marker_length      chem_pop_size (mutate by multiples of 2?)

excite_min = 0
excite_max = 1
connect_prob_min = 0
connect_prob_max = 1
weight_mean_min = 0
weight_mean_max = 0.015
weight_stdev_min = 0
weight_stdev_max = 0.005
delay_mean_min = 0
delay_mean_max = 30
delay_stdev_min = 0
delay_stdev_max = 10
lvl_stop_min = 0    #arbitrary atm
lvl_stop_max = 10   #arbitrary atm
lvl_noise_min = 0   #arbitrary atm
lvl_noise_max = 10  #arbitrary atm
neuron_params = 9
neuron_pop = [[0 for i in range(neuron_params)] for i in range(neuron_pop_size)]
for i in range(neuron_pop_size):
    #excititory probability
    neuron_pop[i][0] = np.random.uniform(excite_min,excite_max)
    #connection probability
    neuron_pop[i][1] = np.random.uniform(connect_prob_min,connect_prob_max)
    #weight mean
    neuron_pop[i][2] = np.random.uniform(weight_mean_min,weight_mean_max)
    #weight stdev
    neuron_pop[i][3] = np.random.uniform(weight_stdev_min,weight_stdev_max)
    #delay mean
    neuron_pop[i][4] = np.random.uniform(delay_mean_min,delay_mean_max)
    #delay stdev
    neuron_pop[i][5] = np.random.uniform(delay_stdev_min,delay_stdev_max)
    #level at which it will stop
    neuron_pop[i][6] = np.random.uniform(lvl_stop_min,lvl_stop_max)
    #noise of level detection
    neuron_pop[i][7] = np.random.uniform(lvl_noise_min,lvl_noise_max)
    #chemical marker
    neuron_pop[i][8] = np.random.randint(0,np.power(2,marker_length))


#chemical gradients
    #decay constant and shape (in each dimention or uniform)
    #attractive/repulsive to certain neuron markers and degree?
    #initial concentration

#decay constant (in distance not time) = 0.1-2 (depends on net_size)
    #decay constant +x (in distance not time) = 0.1-2 (depends on net_size)
    #decay constant -x (in distance not time) = 0.1-2 (depends on net_size)
    #decay constant +y (in distance not time) = 0.1-2 (depends on net_size)
    #decay constant -y (in distance not time) = 0.1-2 (depends on net_size)
#attractive/repulsive marker = 2^marker length
#strength of interaction = 0-10 (arbitrary, need to know other value interactions)
decay_min = 0.2
decay_max = 2
strength_min = 0    #arbitrary atm
strength_max = 15   #arbitrary atm
chem_params = 3
chem_pop = [[0 for i in range(chem_params)] for i in range(chem_pop_size)]
for i in range(chem_pop_size):
    #decay constant
    chem_pop[i][0] = np.random.uniform(decay_min,decay_max)
    #chemical marker
    chem_pop[i][1] = np.random.randint(0,np.power(2,marker_length))
    #stength/ initial concentration
    chem_pop[i][2] = np.random.uniform(strength_min,strength_max)


#2D map orientation - maybe seperate for neurons and chemical
    #position on input and output neural populations
    #postion in the discrete 2D field of neurons
    #postion in the discrete 2D field of checmical secreaters
    #number of seperate chemical secreters
    #number of neural populations
    #size of the field/map
    #no cross breeding between maps of different size or number of 'population'

#map size = 3-10?
#input location = (0-map_size, 0-map_size)
#output location = (0-map_size, 0-map_size)
    #P() of certain neural pop per square = 0-1
    #P() of certain chemical pop per square = 0-1
        #encode location P() by marker, similar to chemical interaction marker
#number of neural populations = ?
#location of neural pops = (x,y)
    #blending into neighbour amount = 0.4 (some random ratio)
#number of chemical populations = ?
#location of chemical pops = (x,y)
    #blending into neighbour amount = 0.4 (some random ratio)
map_size = 4 #keeping fixed for now but in future could be adjustable by the GA
per_cell_min = 20
per_cell_max = 300
max_neuron_types = 5 #keeping fixed for now but in future could be adjustable by the GA
max_chem_types = 5 #keeping fixed for now but in future could be adjustable by the GA
map_params = 2 + (2*max_neuron_types) + (2*max_chem_types)
map_pop = [[0 for i in range(map_params)] for i in range(map_pop_size)]
for i in range(map_pop_size):
    #input location
    map_pop[i][0] = np.random.randint(0,map_size) + (np.random.randint(0,map_size) * 10)
    #output location
    map_pop[i][1] = np.random.randint(0,map_size) + (np.random.randint(0,map_size) * 10)
    #neurons to include
    j = 0
    while j < max_neuron_types:
        map_pop[i][2+(j*2)] = np.random.randint(0,neuron_pop_size)
        map_pop[i][3+(j*2)] = np.random.randint(0,map_size) + (np.random.randint(0,map_size) * 10)
        j += 1
    #chemicals to include
    j = 0
    while j < max_chem_types:
        map_pop[i][2+(max_neuron_types*2)+(j*2)] = np.random.randint(0,chem_pop_size)
        map_pop[i][3+(max_neuron_types*2)+(j*2)] = np.random.randint(0,map_size) + (np.random.randint(0,map_size) * 10)
        j += 1

#test population (all combos of 3 evo properties, or pos not depends on construction)
    #many combinations of ball and beam starting point
    #roll together if time is an issue
    #random initial conditions?
    #random test ordering to build robustness
    #average distance^2 from the centre assuming non random tests

    #build whole chem map, average gradient in the x and y direction

    #x'' = (x'*theta' - g*sin(theta)) / (1 + moi_beam/(mass*radius^2))
    #theta'' = (torque - mass*g*cos(theta) - 2*mass*x*x'*theta') / (mass*x^2 + moi_ball + moi_beam)
    #x is +ve if on the left side and -ve if on the right
    #theta +ve if clockwise -ve if anti-clock


#evolve on property keeping the others fixed
    #select the best and evolve against it
    #or keep a few of the best and evaluate them in combination

#