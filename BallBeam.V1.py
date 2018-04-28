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

#define GA characteristics
number_of_generations = 20
cycles_per_generation = 1
neuron_per_cycle = 1
chem_per_cycle = 1
map_per_cycle = 1

#define the network parameters
neuron_pop_size = 10
chem_pop_size = 10
map_pop_size = 10
input_poisson_min = 2
input_poisson_max = 50
marker_length = 7
map_size = 4 #keeping fixed for now but in future could be adjustable by the GA
per_cell_min = 50
per_cell_max = 1000
max_neuron_types = 7 #2 are in&output - keeping fixed for now but in future could be adjustable by the GA
max_chem_types = 5 #keeping fixed for now but in future could be adjustable by the GA

#define experimental paramters
beam_length = 2 #centred half way
g = 9.81
mass = 0.01
radius = 0.01
moi_ball = 0.0000006
moi_beam = 0.02

left = 0
right = 1
down = 2
up = 3

port_offset = 0

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
lvl_stop_max = 2   #arbitrary atm
lvl_noise_min = 0   #arbitrary atm
lvl_noise_max = 5  #arbitrary atm
neuron_params = 9
neuron_pop = [[0 for i in range(neuron_params)] for j in range(neuron_pop_size)]
for i in range(neuron_pop_size):
    #excititory probability
    excite_index = 0
    neuron_pop[i][excite_index] = np.random.uniform(excite_min,excite_max)
    #connection probability
    connect_prob_index = 1
    neuron_pop[i][connect_prob_index] = np.random.uniform(connect_prob_min,connect_prob_max)
    #weight mean
    weight_mean_index = 2
    neuron_pop[i][weight_mean_index] = np.random.uniform(weight_mean_min,weight_mean_max)
    #weight stdev
    weight_stdev_index = 3
    neuron_pop[i][weight_stdev_index] = np.random.uniform(weight_stdev_min,weight_stdev_max)
    #delay mean
    delay_mean_index = 4
    neuron_pop[i][delay_mean_index] = np.random.uniform(delay_mean_min,delay_mean_max)
    #delay stdev
    delay_stdev_index = 5
    neuron_pop[i][delay_stdev_index] = np.random.uniform(delay_stdev_min,delay_stdev_max)
    #level at which it will stop
    lvl_stop_index = 6
    neuron_pop[i][lvl_stop_index] = np.random.uniform(lvl_stop_min,lvl_stop_max)
    #noise of level detection
    lvl_noise_index = 7
    neuron_pop[i][lvl_noise_index] = np.random.uniform(lvl_noise_min,lvl_noise_max)
    #chemical marker
    chem_marker_index = 8
    neuron_pop[i][chem_marker_index] = np.random.randint(0,np.power(2,marker_length))


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
chem_pop = [[0 for i in range(chem_params)] for j in range(chem_pop_size)]
for i in range(chem_pop_size):
    #decay constant
    chem_pop[i][0] = np.random.uniform(decay_min,decay_max)
    #stength/ initial concentration
    chem_pop[i][1] = np.random.uniform(strength_min,strength_max)
    #chemical marker
    chem_pop[i][2] = np.random.randint(0,np.power(2,marker_length))


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
map_neuron_params = 4
map_chem_params = 3
map_params = 2 + (map_neuron_params*max_neuron_types) + (map_chem_params*max_chem_types)
map_pop = [[0 for i in range(map_params)] for j in range(map_pop_size)]
for i in range(map_pop_size):
    #input poisson characteristics
    input_poisson_low = 0
    input_poisson_high = 1
    map_pop[i][input_poisson_low] = np.random.uniform(input_poisson_min,input_poisson_max)
    map_pop[i][input_poisson_high] = np.random.uniform(input_poisson_min,input_poisson_max)
    #neurons to include
    map_neuron_index = input_poisson_high + 1
    map_neuron_loc_x = map_neuron_index + 1
    map_neuron_loc_y = map_neuron_loc_x + 1
    map_neuron_count = map_neuron_loc_y + 1
    #map_neuron_max_connect_prob = map_neuron_count + 1
    k = 0
    while k < max_neuron_types:
        #select the neuron pop
        map_pop[i][map_neuron_index+(k*map_neuron_params)] = np.random.randint(0,neuron_pop_size)
        #place the neuron pop in the map
        map_pop[i][map_neuron_loc_x+(k*map_neuron_params)] = np.random.randint(0,map_size)
        map_pop[i][map_neuron_loc_y+(k*map_neuron_params)] = np.random.randint(0,map_size)
        #number of neurons at location
        map_pop[i][map_neuron_count+(k*map_neuron_params)] = np.random.randint(per_cell_min,per_cell_max)
        k += 1
    #chemicals to include
    map_chem_index = map_neuron_index+(max_neuron_types*map_neuron_params)
    map_chem_loc_x = map_chem_index + 1
    map_chem_loc_y = map_chem_loc_x + 1
    k = 0
    while k < max_chem_types:
        #select the chem pop
        map_pop[i][map_chem_index+(k*map_chem_params)] = np.random.randint(0,chem_pop_size)
        #place the chem pop
        map_pop[i][map_chem_loc_x+(k*map_chem_params)] = np.random.randint(0,map_size)
        map_pop[i][map_chem_loc_y+(k*map_chem_params)] = np.random.randint(0,map_size)
        k += 1

#return bit string of marker code to marker length bit code
def marker_bits(marker_no):
    bit_string = [0 for i in range(marker_length)]
    bit_string[0] = marker_no%2
    marker_no -= bit_string[0]
    for i in range(1,marker_length):
        #marker_no = marker_no - (np.power(2,(i-1))*bit_string[i-1])
        bit_string[i] = marker_no % np.power(2,i+1)
        if bit_string[i] != 0:
            marker_no -= bit_string[i]
            bit_string[i] = 1
        else:
            bit_string[i] = -1
    return bit_string

#build whole chem map, average gradient in the x and y direction
def gradient_creation(map_agent):
    #first create a map of each chemicals concentrations throughout the map
    concentration_chem_map = [[[0 for i in range(max_chem_types)] for j in range(map_size)] for k in range(map_size)]
    for x in range(map_size):
        for y in range(map_size):
            for k in range(max_chem_types):
                #calculate the concentration (base * e^(-d*lamda))
                chem_select = (k*map_chem_params)
                decay_const = chem_pop[map_pop[map_agent][map_chem_index+chem_select]][0]
                base = chem_pop[map_pop[map_agent][map_chem_index+chem_select]][1]
                distance = np.sqrt(np.power((x-map_pop[map_agent][map_chem_loc_x+chem_select]),2) + np.power((y-map_pop[map_agent][map_chem_loc_y+chem_select]),2))
                concentration_chem_map[x][y][k] = base * np.exp(-1*decay_const*distance)
    #map the combined markers concentration at each point in the map
    marker_chem_map = [[[0 for i in range(marker_length)] for j in range(map_size)] for k in range(map_size)]
    for k in range(max_chem_types):
        chem_select = (k * map_chem_params)
        bit_string = marker_bits(chem_pop[map_pop[map_agent][map_chem_index+chem_select]][2])
        for x in range(map_size):
            for y in range(map_size):
                    for m in range(marker_length):
                        marker_chem_map[x][y][m] += concentration_chem_map[x][y][k] * bit_string[m]
    #calculate the gradient of each markers concentration in all 4 dimensions (maybe 6 later with 3d)
    marker_gradient_map = [[[[0 for h in range(4)] for i in range(marker_length)] for j in range(map_size)] for k in range(map_size)]
    gradient_sign = -1 #-1 -> +ve gradient = going up gradient
    for x in range(map_size):
        for y in range(map_size):
            for m in range(marker_length):
                centre = marker_chem_map[x][y][m]
                if x > 0:
                    marker_gradient_map[x][y][m][left] = (centre - marker_chem_map[x-1][y][m]) * gradient_sign
                else:
                    marker_gradient_map[x][y][m][left] = 0
                if x < map_size-1:
                    marker_gradient_map[x][y][m][right]= (centre - marker_chem_map[x+1][y][m]) * gradient_sign
                else:
                    marker_gradient_map[x][y][m][right] = 0
                if y > 0:
                    marker_gradient_map[x][y][m][down] = (centre - marker_chem_map[x][y-1][m]) * gradient_sign
                else:
                    marker_gradient_map[x][y][m][down] = 0
                if y < map_size-1:
                    marker_gradient_map[x][y][m][up] = (centre - marker_chem_map[x][y+1][m]) * gradient_sign
                else:
                    marker_gradient_map[x][y][m][up] = 0
    return marker_gradient_map

#rolls the probability and sees where a marker will be carried along the gradients
    #add some check to stop going backwards?
def gradient_final_location(marker, gradient_map, noise, stop_lvl, x, y):
    final_loc = [0 for i in range(2)]
    force = stop_lvl + 1
    i = 0
    x_minus1 = x
    y_minus1 = y
    x_minus2 = -10
    y_minus2 = -10
    while force > stop_lvl:
        if i != 0:
            x_minus2 = x_minus1
            y_minus2 = y_minus1
            x_minus1 = x
            y_minus1 = y
        total_left = 0
        total_right = 0
        total_down = 0
        total_up = 0
        for m in range(marker_length):
            total_left += marker[m] * gradient_map[x][y][m][left]
            total_right += marker[m] * gradient_map[x][y][m][right]
            total_down += marker[m] * gradient_map[x][y][m][down]
            total_up += marker[m] * gradient_map[x][y][m][up]
        total_left += np.random.uniform(-noise,noise)
        total_right += np.random.uniform(-noise,noise)
        total_down += np.random.uniform(-noise,noise)
        total_up += np.random.uniform(-noise,noise)
        lateral = total_right - total_left
        vertical = total_up - total_down
        if np.abs(lateral) > stop_lvl or np.abs(vertical) > stop_lvl:
            if np.abs(lateral) > np.abs(vertical):
                if lateral > 0:
                    x += 1
                    if x > map_size - 1:
                        x -= 1
                        break
                else:
                    x -= 1
                    if x < 0:
                        x += 1
                        break
            else:
                if vertical > 0:
                    y += 1
                    if y > map_size - 1:
                        y -= 1
                        break
                else:
                    y -= 1
                    if y < 0:
                        y += 1
                        break
        else:
            break
        #check if it is constantly switching back and forth between locations
        if x == x_minus2 and y == y_minus2:
            #randomly choose one of the 2 locations
            if np.random.uniform(0,1) > 0.5:
                x = x_minus1
            if np.random.uniform(0,1) > 0.5:
                y = y_minus1
            break
        i += 1
        if i > 100:
            if np.random.uniform(0, 1) > 0.9:
                break
    if i > 10:
        print i
    final_loc[0] = x
    final_loc[1] = y
    return final_loc

#function to generate connections of each neuron
def neuron_connect_dist(agent, neuron):
    n_selected = neuron * map_neuron_params
    n_agent = map_pop[agent][map_neuron_index+n_selected]
    n_x = map_pop[agent][map_neuron_loc_x+n_selected]
    n_y = map_pop[agent][map_neuron_loc_y+n_selected]
    n_number = map_pop[agent][map_neuron_count+n_selected]
    n_noise = neuron_pop[n_agent][lvl_noise_index]
    n_stop = neuron_pop[n_agent][lvl_stop_index]
    n_marker = marker_bits(neuron_pop[n_agent][chem_marker_index])
    max_connect_probability = neuron_pop[n_agent][connect_prob_index]
    gradient_map = gradient_creation(agent)
    final_axon_locs = [[0 for i in range(map_size)] for j in range(map_size)]
    for i in range(n_number):
        final_loc = gradient_final_location(n_marker, gradient_map, n_noise, n_stop, n_x, n_y)
        final_axon_locs[final_loc[0]][final_loc[1]] += 1
    for i in range(map_size):
        for j in range(map_size):
            final_axon_locs[i][j] = (final_axon_locs[i][j] / float(n_number)) * max_connect_probability
    return final_axon_locs


#creates the neural network to be placed on SpiNNaker
def create_spinn_net(agent):
    global port_offset
    p.setup(timestep=1.0, min_delay=1, max_delay=(delay_mean_max+(4*delay_stdev_max)))
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 500)
    #initialise the populations
    n_pop_labels = []
    n_pop_list = []
    #initialise the populations
    for i in range(max_neuron_types):
        n_selected = i * map_neuron_params
        n_index = map_pop[agent][map_neuron_index + n_selected]
        n_number = map_pop[agent][map_neuron_count + n_selected]
        #set up the input as a live spike source
        if i == 0:
            n_pop_labels.append("Input_pop{}".format(i))
            n_pop_list.append(p.Population(n_number, p.SpikeSourcePoisson(rate=0), label=n_pop_labels[i]))
            p.external_devices.add_poisson_live_rate_control(n_pop_list[i],database_notify_port_num=16000 + port_offset)
        #set up all other populations
        else:
            n_pop_labels.append("neuron{}-index{}".format(i,n_index))
            n_pop_list.append(p.Population(n_number, p.IF_cond_exp(), label=n_pop_labels[i]))
    #connect the populations
    for i in range(max_neuron_types):
        n_selected = i * map_neuron_params
        n_index = map_pop[agent][map_neuron_index + n_selected]
        n_number = map_pop[agent][map_neuron_count + n_selected]
        excite_prob = neuron_pop[n_index][excite_index]
        connection_prob_dist = neuron_connect_dist(agent, i)

#tests a particular agent on the required configuration of tests
def ball_and_beam_tests(agent, combined, random, number_of_tests, duration):
    if combined == False:
        print 'not combined, reroll every time'
    else:
        create_spinn_net(agent)

# test population (all combos of 3 evo properties, or pos not depends on construction)
    # many combinations of ball and beam starting point
    # roll together if time is an issue
    # random initial conditions?
    # random test ordering to build robustness
    # average distance^2 from the centre assuming non random tests

for gen in range(number_of_generations):
    for agent in range(map_pop_size):
        print 'starting agent {}'.format(agent)
        fitness = ball_and_beam_tests(agent, True, False, 8, 5)

#Test the population
    #x'' = (x'*theta' - g*sin(theta)) / (1 + moi_beam/(mass*radius^2))
    #theta'' = (torque - mass*g*cos(theta) - 2*mass*x*x'*theta') / (mass*x^2 + moi_ball + moi_beam)
    #x is +ve if on the left side and -ve if on the right
    #theta +ve if clockwise -ve if anti-clock


#evolve on property keeping the others fixed
    #select the best and evolve against it
    #or keep a few of the best and evaluate them in combination

#

print 'DONE!!!'