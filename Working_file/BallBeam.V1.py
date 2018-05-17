import spynnaker8 as p
# from pyNN.utility.plotting import Figure, Panel
import time
import copy
# import pylab
import numpy as np
import threading
# from threading import Condition
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
from pyNN.random import RandomDistribution# as rand
#import spynnaker8.spynakker_plotting as splot
# import csv
# import pandas

#define GA characteristics
number_of_generations = 20
cycles_per_generation = 1
neuron_per_cycle = 1
chem_per_cycle = 1
base_probability = 1 # the base number each neuron/chemical agent is given to influence it's chance of being chosen
re_roll = False # whether a chemical marker is simply re-rolled or each bit is individually mutated
map_per_cycle = 3
elitism = 0.5
selecting_scale = 10
mutation_rate = 0.01
crossover = 0.5
stdev_range = 0.15

#define the network parameters
neuron_pop_size = 150
chem_pop_size = 100
map_pop_size = 20
input_poisson_min = 1
input_poisson_max = 20
thread_input = False
held_input_pop_size = 0
marker_length = 7
map_size = 4 #keeping fixed for now but in future could be adjustable by the GA
per_cell_min = 10
per_cell_max = 100
#input comprised of:
    #position of the ball
    #angle of the beam
        #velocity of the ball
        #velocity of the beam
input_labels = list()
output_labels = list()
no_input_pop = 2
no_output_pop = 2
max_neuron_types = 5 + no_input_pop + no_output_pop #keeping fixed for now but in future could be adjustable by the GA
max_chem_types = 5 #keeping fixed for now but in future could be adjustable by the GA

#define experimental paramters
number_of_tests = 2
time_scale_factor = 1
duration = 4000 #ms
duration *= time_scale_factor
time_segments = 200 #duration of a segment
time_segments *= time_scale_factor
average_runtime = 0.1 #time it takes to complete the setting of poisson variables
fitness_begin = 0 #segment when fitness calculations begin
experimental_record = []
beam_length = 2 #centred half way
off_the_beam = -np.inf #fitness punishment if it falls off the beam
g = 9.81
mass = 1e-3
radius = 0.01
moi_ball = 6e-7
moi_beam = 2e-2
max_angle = np.pi/4
min_angle = -max_angle
current_agent = 0
current_angle = 0
current_position = 0
current_beam_vlct= 0
current_ball_vlct = 0
current_beam_acc = 0
current_ball_acc = 0
spike_to_torque = 0.005
starting_position_min = 0.1 #ratio of the length from the centre
starting_position_max = 0.8
starting_angle_max = 1
varying = "position" #position, angle or both
fitness_calculation = "linear" #"linear", "quadratic" or within boundary "time"
fitness_boundary = 0.05 #where it has to be placed within to constitute controlled positioning

left = 0
right = 1
down = 2
up = 3

port_offset = 0
running_status = False
live_connection = p.external_devices.SpynnakerLiveSpikesConnection(local_port=(1800 + port_offset))
poisson_control = p.external_devices.SpynnakerPoissonControlConnection(poisson_labels=input_labels,
                                                                           local_port=1600 + port_offset)

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
connect_prob_max = 0.5
weight_mean_min = 0
weight_mean_max = 0.01
weight_stdev_min = 0
weight_stdev_max = 0.003
delay_mean_min = 2
delay_mean_max = 40
delay_stdev_min = 0
delay_stdev_max = 5
delay_cap = delay_mean_max + (3*delay_stdev_max)
lvl_stop_min = 0    #arbitrary atm
lvl_stop_max = 2   #arbitrary atm
lvl_noise_min = 0   #arbitrary atm
lvl_noise_max = 5  #arbitrary atm
neuron_params = 9
temp_neuron_pop_size = int(round(neuron_pop_size * (1 - elitism)))
temp_neuron_pop = [[0 for i in range(neuron_params)] for j in range(temp_neuron_pop_size)]
neuron_pop = [[0 for i in range(neuron_params)] for j in range(neuron_pop_size+1)]
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
temp_chem_pop_size = int(round(chem_pop_size * (1 - elitism)))
temp_chem_pop = [[0 for i in range(chem_params)] for j in range(temp_chem_pop_size)]
chem_pop = [[0 for i in range(chem_params)] for j in range(chem_pop_size+1)]
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
map_params = (2*no_input_pop) + (map_neuron_params*max_neuron_types) + (map_chem_params*max_chem_types)
temp_map_pop_size = int(round(map_pop_size * (1 - elitism)))
temp_map_pop = [[0 for i in range(map_params)] for j in range(temp_map_pop_size)]
map_pop = [[0 for i in range(map_params)] for j in range(map_pop_size+1)]
for i in range(map_pop_size):
    #input poisson characteristics
    input_poisson_low = 0
    input_poisson_high = 1
    for j in range(no_input_pop):
        map_pop[i][input_poisson_low + (j*no_input_pop)] = np.random.uniform(input_poisson_min, input_poisson_max)
        map_pop[i][input_poisson_high + (j*no_input_pop)] = np.random.uniform(input_poisson_min, input_poisson_max)
    #neurons to include
    map_neuron_index = (2 * no_input_pop)
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
    if bit_string[0] == 0:
        bit_string[0] = -1
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

#function to keep a record of each output pops firing
motor_spikes = [0 for i in range(no_output_pop)]
def receive_spikes(label, time, neuron_ids):
    for neuron_id in neuron_ids:
        for i in range(no_output_pop):
            if label == output_labels[i]:
                motor_spikes[i] += 1
                #print "\n\nmotor {} - time = {}\n\n".format(i, time)

#adjusts the poisson inputs live during a run
def poisson_setting(label, connection):
    global current_angle
    global current_beam_vlct
    global current_beam_acc
    global current_position
    global current_ball_vlct
    global current_ball_acc
    global current_agent
    global average_runtime
    global motor_spikes
    #convert to poisson rate
    min_poisson_dist = map_pop[current_agent][input_poisson_low]
    max_poisson_dist = map_pop[current_agent][input_poisson_high]
    min_poisson_angle = map_pop[current_agent][input_poisson_low+no_input_pop]
    max_poisson_angle = map_pop[current_agent][input_poisson_high+no_input_pop]
    float_time = float(time_segments - (average_runtime * 1000)) / 1000
    total = 0
    # experimental_record.append([current_position, current_ball_vlct,current_ball_acc,
    #                             current_angle, current_beam_vlct, current_beam_acc, time.clock()])
    #loop through for the duration of a run
    start = time.clock()
    for i in range(0, duration, time_segments):
        #set at the precise time needed
        float_time = max((float(i) / 1000.0) - (time.clock() - start), 0)
        time.sleep(float_time)
        #translate motor commands into movement of the beam and the ball
            # x'' = (x'*theta'^2 - g*sin(theta)) / (1 + moi_ball/(mass*radius^2))
            # theta'' = (torque - mass*g*x*cos(theta) - 2*mass*x*x'*theta') / (mass*x^2 + moi_ball + moi_beam)
            # x is +ve if on the left side and -ve if on the right
            # theta +ve if clockwise -ve if anti-clock
            #alternate equations
            #(Ib + ms * rb^2) * thetab'' + (2 * ms * rb * rb' * thetab') + (ms * g * rb * costhetab) = torque
            #thetab'' = (torque - (ms * g * rb * costhetab) - (2 * ms * rb * rb' * thetab')) / (Ib + ms * rb^2)
            #rb'' = -5/7 * (g * sinthetab - rb * thetab^2)
        clockwise = 0
        anticlock = 0
        for j in range(no_output_pop):
            #clockwise rotation
            if j < no_output_pop / 2:
                clockwise += motor_spikes[j]
                motor_spikes[j] = 0
            #anticlockwise
            else:
                anticlock += motor_spikes[j]
                motor_spikes[j] = 0
        total_clock = clockwise - anticlock
        torque = float(total_clock) * spike_to_torque
        current_ball_acc = (current_ball_vlct*np.power(current_beam_vlct,2)) - (g*np.sin(current_angle))
        current_ball_acc /= (1 + (moi_ball/(mass*np.power(radius,2))))
        # current_ball_acc = (-5.0/7.0) * ((g * np.sin(current_angle)) - (current_position * np.power(current_beam_vlct, 2)))
        current_beam_acc = (torque - (mass*g*current_position*np.cos(current_angle)) -
                            (2*mass*current_position*current_ball_vlct*current_beam_vlct))
        #current_beam_acc = torque
        current_beam_acc /= (mass*np.power(current_position,2)) + moi_ball + moi_beam
        seconds_window = float(time_segments / 1000.0)
        current_ball_vlct += float(current_ball_acc * seconds_window)
        current_position += float(current_ball_vlct *seconds_window)
        current_beam_vlct += float(current_beam_acc * seconds_window)
        #check that the beam is not at the maximum angle
        previous_angle = current_angle
        current_angle += float(current_beam_vlct * seconds_window)
        current_angle = max(min(current_angle, max_angle), min_angle)
        if current_angle == previous_angle:
            current_beam_acc = 0
            current_beam_vlct = 0
        print "clock = {}\tanti = {}\ttorque = {}\tpos = {}\tangle = {}".format(clockwise,anticlock,torque,current_position,current_angle)
        #set poisson rate
        current_pos_ratio = max(min((current_position + beam_length) / (beam_length * 2), 1), 0)
        poisson_position = min_poisson_dist + ((max_poisson_dist - min_poisson_dist) * current_pos_ratio)
        current_ang_ratio = (current_angle + max_angle) / (max_angle * 2)
        poisson_angle = min_poisson_angle + ((max_poisson_angle - min_poisson_angle) * current_ang_ratio)
        #print "\tpoisson angle = {}\tposition = {}".format(poisson_angle, poisson_position)
        n_number = map_pop[agent][map_neuron_count]
        #set at the precise time needed
        # time.sleep(max((float(i) / 1000.0) - (time.clock() - start), 0))
        connection.set_rates(input_labels[0], [(i, int(round(poisson_position))) for i in range(n_number)])
        n_number = map_pop[agent][map_neuron_count+map_neuron_params]
        connection.set_rates(input_labels[1], [(i, int(round(poisson_angle))) for i in range(n_number)])
        experimental_record.append([current_position, current_ball_vlct,current_ball_acc,
                                    current_angle, current_beam_vlct, current_beam_acc, time.clock()])
        if abs(current_position) > beam_length or time.clock() - start > float(duration) / 1000.0:
            #p.end()
            break
        finish = time.clock()
        print "elapsed time = {}\t{} - {}\tave_float = {}".format(finish - start, finish, start, float_time)
    #print 'total = {}, average = {}'.format(total, total/len(experimental_record))

#the function which sets the poisson inputs for the threading case
def threading_setting(connection, start):
    new_start = time.clock()
    if new_start - start < float(duration) / 1000.0:
        print "\nset the rate at ", new_start - start
        global current_angle
        global current_beam_vlct
        global current_beam_acc
        global current_position
        global current_ball_vlct
        global current_ball_acc
        global current_agent
        global average_runtime
        global motor_spikes
        global experimental_record
        # convert to poisson rate
        min_poisson_dist = map_pop[current_agent][input_poisson_low]
        max_poisson_dist = map_pop[current_agent][input_poisson_high]
        min_poisson_angle = map_pop[current_agent][input_poisson_low + no_input_pop]
        max_poisson_angle = map_pop[current_agent][input_poisson_high + no_input_pop]
        clockwise = 0
        anticlock = 0
        for j in range(no_output_pop):
            # clockwise rotation
            if j < no_output_pop / 2:
                clockwise += motor_spikes[j]
                motor_spikes[j] = 0
            # anticlockwise
            else:
                anticlock += motor_spikes[j]
                motor_spikes[j] = 0
        total_clock = clockwise - anticlock
        torque = float(total_clock) * spike_to_torque
        current_ball_acc = (current_ball_vlct * np.power(current_beam_vlct, 2)) - (g * np.sin(current_angle))
        current_ball_acc /= (1 + (moi_ball / (mass * np.power(radius, 2))))
        current_beam_acc = (torque - (mass * g * current_position * np.cos(current_angle)) -
                            (2 * mass * current_position * current_ball_vlct * current_beam_vlct))
        # current_beam_acc = torque
        current_beam_acc /= (mass * np.power(current_position, 2)) + moi_ball + moi_beam
        seconds_window = float(time_segments / 1000.0)
        current_ball_vlct += float(current_ball_acc * seconds_window)
        current_position += float(current_ball_vlct * seconds_window)
        current_beam_vlct += float(current_beam_acc * seconds_window)
        previous_angle = current_angle
        current_angle += float(current_beam_vlct * seconds_window)
        current_angle = max(min(current_angle, max_angle), min_angle)
        if current_angle == previous_angle:
            current_beam_acc = 0
            current_beam_vlct = 0
        print "clock = {}\tanti = {}\ttorque = {}\tpos = {}\tangle = {}".format(clockwise, anticlock, torque,
                                                                                current_position, current_angle)
        # set poisson rate
        current_pos_ratio = max(min((current_position + beam_length) / (beam_length * 2), 1), 0)
        poisson_position = min_poisson_dist + ((max_poisson_dist - min_poisson_dist) * current_pos_ratio)
        current_ang_ratio = (current_angle + max_angle) / (max_angle * 2)
        poisson_angle = min_poisson_angle + ((max_poisson_angle - min_poisson_angle) * current_ang_ratio)
        #print "setting @ {}\tpois ang = {}\tpois pos = {}".format(time.clock(), poisson_angle, poisson_position)
        n_number = map_pop[agent][map_neuron_count]
        connection.set_rates(input_labels[0], [(i,int(round(poisson_position))) for i in range(n_number)])
        n_number = map_pop[agent][map_neuron_count + map_neuron_params]
        connection.set_rates(input_labels[1], [(i,int(round(poisson_angle))) for i in range(n_number)])
        experimental_record.append([current_position, current_ball_vlct, current_ball_acc,
                                    current_angle, current_beam_vlct, current_beam_acc, time.clock()])
        print "finished setting rate at {}, taking {}".format(time.clock() - start, time.clock() - new_start)

#recursively call the setting function
def poisson_thread_recursion(connection, start, iteration):
    global current_position
    time.sleep(max((float(iteration)*(float(time_segments)/1000.0)) - (time.clock() - start), 0))
    threading.Thread(threading_setting(connection, start)).start()
    if time.clock() - start < float(duration) / 1000.0 and abs(current_position) < beam_length:
        poisson_thread_recursion(connection, start, iteration + 1)

#adjusts the poisson inputs using timer threads during a run
def poisson_threading(label, connection):
    start = time.clock()
    threading.Thread(threading_setting(connection, start)).start()
    poisson_thread_recursion(connection, start, 1)
    finish = time.clock()
    print "\nelapsed time = {}\t{} - {}".format(finish - start, finish, start)

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
        total_left += np.random.normal(0,noise)
        total_right += np.random.normal(0,noise)
        total_down += np.random.normal(0,noise)
        total_up += np.random.normal(0,noise)
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

#function maps which neuron pop a neuron pop connects to
def neuron2neuron(agent, neuron):
    connect_dist = neuron_connect_dist(agent, neuron)
    all_nxy = [[0 for i in range(2)] for j in range(max_neuron_types)]
    a2b = [0 for i in range(max_neuron_types)]
    for i in range(max_neuron_types):
        all_nxy[i][0] = map_pop[agent][map_neuron_loc_x+(i*map_neuron_params)]
        all_nxy[i][1] = map_pop[agent][map_neuron_loc_y+(i*map_neuron_params)]
        a2b[i] = connect_dist[all_nxy[i][0]][all_nxy[i][1]]
    return a2b

#creates the neural network to be placed on SpiNNaker
def create_spinn_net(agent):
    global port_offset
    global live_connection
    global poisson_control
    global input_labels
    global output_labels
    p.setup(timestep=1.0, min_delay=1, max_delay=(delay_mean_max+(4*delay_stdev_max)))
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 64)
    #initialise the variable and reset if multiple run
    n_pop_labels = []
    n_pop_list = []
    n_proj_list = []
    if port_offset != 0:
        for i in range(no_output_pop):
            del output_labels[0]
        for i in range(no_input_pop):
            del input_labels[0]
    #initialise the populations
    total_n = 0
    for i in range(max_neuron_types):
        n_selected = i * map_neuron_params
        n_index = map_pop[agent][map_neuron_index + n_selected]
        n_number = map_pop[agent][map_neuron_count + n_selected]
        total_n += n_number
        #set up the input as a live spike source
        if i < no_input_pop:
            if held_input_pop_size != 0:
                map_pop[agent][map_neuron_count + n_selected] = held_input_pop_size
                n_number = map_pop[agent][map_neuron_count + n_selected]
            n_pop_labels.append("Input_pop{}/{}-index{}".format(i, i, n_index))
            input_labels.append("Input_pop{}/{}-index{}".format(i, i, n_index))
            n_pop_list.append(p.Population(n_number, p.SpikeSourcePoisson(rate=input_poisson_min), label=n_pop_labels[i]))
            p.external_devices.add_poisson_live_rate_control(n_pop_list[i],database_notify_port_num=16000 + port_offset)
        #set up output pop
        elif i < no_input_pop + no_output_pop:
            n_pop_labels.append("Output_pop{}/{}-index{}".format(i-no_input_pop, i, n_index))
            output_labels.append("Output_pop{}/{}-index{}".format(i-no_input_pop, i, n_index))
            n_pop_list.append(p.Population(n_number, p.IF_cond_exp(), label=n_pop_labels[i]))
            p.external_devices.activate_live_output_for(
                n_pop_list[i], database_notify_port_num=18000 + port_offset, port=14000 + port_offset)
        #set up all other populations
        else:
            n_pop_labels.append("neuron{}-index{}".format(i,n_index))
            n_pop_list.append(p.Population(n_number, p.IF_cond_exp(), label=n_pop_labels[i]))
        # n_pop_list[i].record(["spikes"])
    print "all neurons in the network = ", total_n

    poisson_control = p.external_devices.SpynnakerPoissonControlConnection(poisson_labels=input_labels,
                                                                           local_port=16000 + port_offset)
    if thread_input == False:
        poisson_control.add_start_callback(n_pop_list[0].label, poisson_setting)
        # poisson_control.add_start_callback(n_pop_list[1].label, poisson_setting)
    else:
        poisson_control.add_start_callback(n_pop_list[0].label, poisson_threading)
        # poisson_control.add_start_callback(n_pop_list[1].label, poisson_threading)
    live_connection = p.external_devices.SpynnakerLiveSpikesConnection(
        receive_labels=output_labels, local_port=(18000 + port_offset))
    # live_connection = p.external_devices.SpynnakerLiveSpikesConnection(
    #     receive_labels=output_labels, local_port=(18000 + port_offset))
    for i in range(no_output_pop):
        live_connection.add_receive_callback(n_pop_labels[no_input_pop+i], receive_spikes)
    #connect the populations
    for i in range(max_neuron_types):
        n_selected = i * map_neuron_params
        n_index = map_pop[agent][map_neuron_index + n_selected]
        #n_number = map_pop[agent][map_neuron_count + n_selected]
        excite_prob = neuron_pop[n_index][excite_index]
        weight_mu = neuron_pop[n_index][weight_mean_index]
        weight_sdtev = neuron_pop[n_index][weight_stdev_index]
        delay_mu = neuron_pop[n_index][delay_mean_index]
        delay_sdtev = neuron_pop[n_index][delay_stdev_index]
        connection_list = neuron2neuron(agent, i)
        #moderate the connection probability based excite probability
        for j in range(no_input_pop, max_neuron_types):
            if connection_list[j] > 1e-6:
                con_excite_prob = connection_list[j] * excite_prob
                con_inhib_prob = connection_list[j] * (1 - excite_prob)
                print "{}\t{}".format(con_excite_prob, con_inhib_prob)
                print "weight mu = {}\t stdev = {}".format(weight_mu, weight_sdtev)
                print "delay mu = {}\t stdev = {}".format(delay_mu, delay_sdtev)
                weights = RandomDistribution("normal_clipped", mu=weight_mu, sigma=weight_sdtev, low=0, high=np.inf)
                delays = RandomDistribution("normal_clipped", mu=delay_mu, sigma=delay_sdtev, low=1, high=delay_cap)
                synapse = p.StaticSynapse(weight=weights, delay=delays)
                n_proj_list.append(p.Projection(n_pop_list[i], n_pop_list[j],
                                                p.FixedProbabilityConnector(con_excite_prob),
                                                synapse, receptor_type="excitatory"))
                n_proj_list.append(p.Projection(n_pop_list[i], n_pop_list[j],
                                                p.FixedProbabilityConnector(con_inhib_prob),
                                                synapse, receptor_type="inhibitory"))

def seperate_test(agent, angle, distance, continuous):
    global current_angle
    global current_beam_vlct
    global current_beam_acc
    global current_position
    global current_ball_vlct
    global current_ball_acc
    global current_agent
    global running_status
    #reset and run
    current_angle = angle
    current_beam_vlct = 0
    current_beam_acc = 0
    current_position = distance
    current_ball_vlct = 0
    current_ball_acc = 0
    current_agent = agent
    if continuous == False:
        p.reset()
    for i in range(no_output_pop):
        motor_spikes[i] = 0
    list_length = len(experimental_record)
    for i in range(list_length):
        del experimental_record[0]
    experimental_record.append([current_position, current_ball_vlct,current_ball_acc,
                                current_angle, current_beam_vlct, current_beam_acc, time.clock()])
    running_status = True
    p.run(duration)
    running_status = False
    #disect experiemntal data
    experiment_length = len(experimental_record)
    running_fitness = 0
    for i in range(fitness_begin, experiment_length):
        if fitness_calculation == "linear":
            running_fitness -= abs(experimental_record[i][0])
        elif fitness_calculation == "quadratic":
            running_fitness -= np.power(experimental_record[i][0], 2)
        else:
            print "timed"
            #calculate how long it took to stablise within a boundary
        if abs(experimental_record[i][0]) > beam_length:
            running_fitness = i
            if off_the_beam > 100000000:
                break
    if running_fitness < 0:
        running_fitness /= float(experiment_length-fitness_begin)
    return running_fitness

def seperate_the_tests(agent, random, continuous):
    global live_connection
    global poisson_control
    overall_fitness = 0
    test_fitness = [0 for i in range(number_of_tests + 1)]
    index = 0
    if random == False:
        if varying == "position":
            segments = number_of_tests / 2
            if segments == 1:
                distance_segment = 0
            else:
                distance_segment = (beam_length * (starting_position_max - starting_position_min)) / (segments - 1)
            for i in range(segments):
                test_fitness[index] = seperate_test(agent, 0, (beam_length*starting_position_max)-(distance_segment * i), continuous)
                if test_fitness[index] < 0:
                    overall_fitness += test_fitness[index]
                index += 1
            for i in range(segments):
                test_fitness[index] = seperate_test(agent, 0, -(beam_length*starting_position_max)+(distance_segment * i), continuous)
                if test_fitness[index] < 0:
                    overall_fitness += test_fitness[index]
                index += 1
        elif varying == "angle":
            angle_segment = ((min_angle - max_angle) * starting_angle_max) / (number_of_tests - 1)
            first_postion = min_angle + ((min_angle - max_angle) * (1-starting_angle_max)) #wrong
            for i in range(number_of_tests):
                test_fitness[i] = seperate_test(agent, first_postion+(angle_segment*i), beam_length*starting_position_max, continuous)
                if test_fitness[i] < 0:
                    overall_fitness += test_fitness[i]
        else:
            print 'trying to vary both'
    else:
        for i in range(number_of_tests):
            random_distance = np.random.uniform(starting_position_min, starting_position_max) * beam_length
            if np.random.uniform(0,1) < 0.5:
                random_distance *= -1
            random_angle = np.random.uniform(min_angle, max_angle)
            test_fitness[i] = seperate_test(agent, random_angle, random_distance, continuous)
            if test_fitness[i] < 0:
                overall_fitness += test_fitness[i]
    test_fitness[number_of_tests] = overall_fitness
    #maybe change these to instantiations of p.external devices instead as that's global?
    poisson_control.close()
    live_connection.close()
    live_connection._handle_possible_rerun_state()
    p.end()
    return test_fitness

#tests a particular agent on the required configuration of tests
def ball_and_beam_tests(agent, combined, random, continuous):
    if combined == False:
        print 'not combined, reroll every time'
    else:
        create_spinn_net(agent)
        agent_data = seperate_the_tests(agent, random, continuous)
    return agent_data

# test population (all combos of 3 evo properties, or pos not depends on construction)
    # many combinations of ball and beam starting point
    # roll together if time is an issue
    # random initial conditions?
    # random test ordering to build robustness
    # average distance^2 from the centre assuming non random tests

#sort and return a ranking based on multiple criteria
def bubble_sort(criteria_1, criteria_2, criteria_3):
    length = len(criteria_1)
    order = [i for i in range(length)]
    equal_index = [[0 for i in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length - i - 1):
            if criteria_1[order[j]] < criteria_1[order[j+1]]:
                temp = order[j+1]
                order[j+1] = order[j]
                order[j] = temp
            elif criteria_1[order[j]] == criteria_1[order[j+1]]:
                if criteria_2[order[j]] < criteria_2[order[j+1]]:
                    temp = order[j+1]
                    order[j+1] = order[j]
                    order[j] = temp
                elif criteria_2[order[j]] == criteria_2[order[j+1]]:
                    if criteria_3[order[j]] < criteria_3[order[j+1]]:
                        temp = order[j+1]
                        order[j+1] = order[j]
                        order[j] = temp
    for i in range(length):
        for j in range(length):
            if criteria_1[order[i]] == criteria_1[order[j]]:
                if criteria_2[order[i]] == criteria_2[order[j]]:
                    if criteria_3[order[i]] == criteria_3[order[j]]:
                        equal_index[order[i]][order[j]] = 1
                        equal_index[order[i]][order[j]] = 1
    i = 0
    while i < length:
        similarities = 0
        listed_equals = []
        for j in range(length):
            if equal_index[order[i]][order[j]] == 1:
                similarities += 1
                listed_equals.append(order[j])
        if similarities != 1:
            for j in range(similarities):
                rand_int = np.random.randint(0, similarities-j)
                order[i+j] = listed_equals[rand_int]
                del listed_equals[rand_int]
        i += similarities
    return order

#compute the fitnesses passed in and rank accordingly
def rank_fitnesses(mutatable, by_rank):
    global fitnesses
    if mutatable == "map":
        number_of_successes = [number_of_tests for i in range(map_pop_size)]
        total_fail_time = [0 for i in range(map_pop_size)]
        total_fitness = [0 for i in range(map_pop_size)]
        for i in range(map_pop_size):
            total_fitness[i] = fitnesses[0][number_of_tests]
            for j in range(number_of_tests):
                if fitnesses[0][j] > 0:
                    number_of_successes[i] -= 1
                    total_fail_time[i] += fitnesses[0][j]
            del fitnesses[0]
        order = bubble_sort(number_of_successes, total_fail_time, total_fitness)
    elif mutatable == "chem":
        chem_type_counter = [base_probability for i in range(chem_pop_size)]
        for i in range(map_pop_size):
            for j in range(max_chem_types):
                chem_type = map_pop[i][map_chem_index + (j * map_chem_params)]
                chem_type_counter[chem_type] += 1
        if by_rank == True:
            order = bubble_sort(chem_type_counter, chem_type_counter, chem_type_counter)
        else:
            order = chem_type_counter
    else:
        neuron_type_counter = [base_probability for i in range(neuron_pop_size)]
        for i in range(map_pop_size):
            for j in range(max_neuron_types):
                neuron_type = map_pop[i][map_neuron_index + (j * map_neuron_params)]
                neuron_type_counter[neuron_type] += 1
        if by_rank == True:
            order = bubble_sort(neuron_type_counter, neuron_type_counter, neuron_type_counter)
        else:
            order = neuron_type_counter
    return order

#select a parent based on some metric
def select_parent(mutatable, not_allowed, prob_dist):
    if mutatable == "map":
        #possibly change it to a non linear prob_dist later but for now pointless (linear)
        total = 0
        not_allowed = (map_pop_size - 1) - not_allowed
        for i in range(map_pop_size):
            if i != not_allowed:
                total += (i+1) * prob_dist
        number = np.random.uniform(total)
        while number > 0:
            if i != not_allowed:
                number -= (i+1) * prob_dist
            i -= 1
        parent = map_pop_size - (i + 2)
    elif mutatable == "chem":
        total = 0
        for i in range(chem_pop_size):
            if i != not_allowed:
                total += prob_dist[i]
        number = np.random.uniform(total)
        while number > 0:
            if i != not_allowed:
                number -= prob_dist[i]
            i -= 1
        parent = i+1
    else:
        total = 0
        for i in range(neuron_pop_size):
            if i != not_allowed:
                total += prob_dist[i]
        number = np.random.uniform(total)
        while number > 0:
            if i != not_allowed:
                number -= prob_dist[i]
            i -= 1
        parent = i+1
    return parent

#mate the map agents
def mate_maps(parent1, parent2):
    child = map_pop_size
    parent = 0
    for i in range(map_params):
        if np.random.uniform(0,1) < crossover:
            parent += 1
        if np.power(-1, parent) == 1:
            map_pop[child][i] = map_pop[parent1][i]
        else:
            map_pop[child][i] = map_pop[parent2][i]
        if np.random.uniform(0,1) < mutation_rate:
            if i < map_neuron_index:
                stdev = stdev_range * (input_poisson_max - input_poisson_min)
                map_pop[child][i] += np.random.normal(0, stdev)
                if map_pop[child][i] >= input_poisson_max:
                    map_pop[child][i] -= (input_poisson_max - input_poisson_min)
                if map_pop[child][i] < input_poisson_min:
                    map_pop[child][i] += (input_poisson_max - input_poisson_min)
            elif i < map_chem_index:
                if (i - map_neuron_index) % map_neuron_params == 0:
                    stdev = stdev_range * (neuron_pop_size)
                    map_pop[child][i] += int(round(np.random.normal(0, stdev)))
                    if map_pop[child][i] >= neuron_pop_size:
                        map_pop[child][i] -= neuron_pop_size
                    if map_pop[child][i] < 0:
                        map_pop[child][i] += neuron_pop_size
                elif (i - map_neuron_index) % map_neuron_params < 3:
                    stdev = stdev_range * (map_size)
                    map_pop[child][i] += int(round(np.random.normal(0, stdev)))
                    if map_pop[child][i] >= map_size:
                        map_pop[child][i] -= map_size
                    if map_pop[child][i] < 0:
                        map_pop[child][i] += map_size
                else:
                    stdev = stdev_range * (per_cell_max - per_cell_min)
                    map_pop[child][i] += int(round(np.random.normal(0, stdev)))
                    if map_pop[child][i] >= per_cell_max:
                        map_pop[child][i] -= (per_cell_max - per_cell_min)
                    if map_pop[child][i] < per_cell_min:
                        map_pop[child][i] += (per_cell_max - per_cell_min)
            else:
                if (i - map_chem_index) % map_chem_params == 0:
                    stdev = stdev_range * (chem_pop_size)
                    map_pop[child][i] += int(round(np.random.normal(0, stdev)))
                    if map_pop[child][i] >= chem_pop_size:
                        map_pop[child][i] -= chem_pop_size
                    if map_pop[child][i] < 0:
                        map_pop[child][i] += chem_pop_size
                else:
                    stdev = stdev_range * (map_size)
                    map_pop[child][i] += int(round(np.random.normal(0, stdev)))
                    if map_pop[child][i] >= map_size:
                        map_pop[child][i] -= map_size
                    if map_pop[child][i] < 0:
                        map_pop[child][i] += map_size

#mate the neuron agents
def mate_neurons(parent1, parent2):
    child = neuron_pop_size
    parent = 0
    for i in range(neuron_params):
        if np.random.uniform(0,1) < crossover:
            parent += 1
        if np.power(-1, parent) == 1:
            neuron_pop[child][i] = neuron_pop[parent1][i]
        else:
            neuron_pop[child][i] = neuron_pop[parent2][i]
        if np.random.uniform(0,1) < mutation_rate:
            if i == excite_index:
                full_range = (excite_max - excite_min)
                maximum = excite_max
                minimum = excite_min
            elif i == connect_prob_index:
                full_range = (connect_prob_max - connect_prob_min)
                maximum = connect_prob_max
                minimum = connect_prob_min
            elif i == weight_mean_index:
                full_range = (weight_mean_max - weight_mean_min)
                maximum = weight_mean_max
                minimum = weight_mean_min
            elif i == weight_stdev_index:
                full_range = (weight_stdev_max - weight_stdev_min)
                maximum = weight_stdev_max
                minimum = weight_stdev_min
            elif i == delay_mean_index:
                full_range = (delay_mean_max - delay_mean_min)
                maximum = delay_mean_max
                minimum = delay_mean_min
            elif i == delay_stdev_index:
                full_range = (delay_stdev_max - delay_stdev_min)
                maximum = delay_stdev_max
                minimum = delay_stdev_min
            elif i == lvl_stop_index:
                full_range = (lvl_stop_max - lvl_stop_min)
                maximum = lvl_stop_max
                minimum = lvl_stop_min
            elif i == lvl_noise_index:
                full_range = (lvl_noise_max - lvl_noise_min)
                maximum = lvl_noise_max
                minimum = lvl_noise_min
            elif i == chem_marker_index:
                if re_roll == True:
                    neuron_pop[child][i] = np.random.randint(0, np.power(2, marker_length))
                else:
                    print "going to reroll them later"
            else:
                print "fuck"
            if i != chem_marker_index:
                stdev = stdev_range * full_range
                neuron_pop[child][i] += np.random.normal(0, stdev)
                if neuron_pop[child][i] >= maximum:
                    neuron_pop[child][i] -= full_range
                if neuron_pop[child][i] < minimum:
                    neuron_pop[child][i] += full_range
    if re_roll == False:
        for i in range(marker_length):
            bit_string = marker_bits(neuron_pop[child][chem_marker_index])
            if np.random.uniform(0, 1) < mutation_rate:
                if bit_string[i] == 1:
                    neuron_pop[child][chem_marker_index] -= np.power(2, i)
                else:
                    neuron_pop[child][chem_marker_index] += np.power(2, i)
        if neuron_pop[child][chem_marker_index] >= np.power(2, marker_length) or neuron_pop[child][chem_marker_index] < 0:
            print "fuck"

#mate the chemical agents
def mate_chemicals(parent1, parent2):
    child = chem_pop_size
    parent = 0
    for i in range(chem_params):
        if np.random.uniform(0, 1) < crossover:
            parent += 1
        if np.power(-1, parent) == 1:
            chem_pop[child][i] = chem_pop[parent1][i]
        else:
            chem_pop[child][i] = chem_pop[parent2][i]
        if np.random.uniform(0, 1) < mutation_rate:
            if i == 0:
                full_range = (decay_max - decay_min)
                maximum = decay_max
                minimum = decay_min
            elif i == 1:
                full_range = (strength_max - strength_min)
                maximum = strength_max
                minimum = strength_min
            elif i == 2:
                if re_roll == True:
                    chem_pop[child][i] = np.random.randint(0, np.power(2, marker_length))
                else:
                    print "going to reroll them later"
            else:
                print "fuck"
            if i != 2:
                stdev = stdev_range * full_range
                chem_pop[child][i] += np.random.normal(0, stdev)
                if chem_pop[child][i] >= maximum:
                    chem_pop[child][i] -= full_range
                if chem_pop[child][i] < minimum:
                    chem_pop[child][i] += full_range
    if re_roll == False:
        for i in range(marker_length):
            bit_string = marker_bits(chem_pop[child][2])
            if np.random.uniform(0, 1) < mutation_rate:
                if bit_string[i] == 1:
                    chem_pop[child][2] -= np.power(2, i)
                else:
                    chem_pop[child][2] += np.power(2, i)
        if chem_pop[child][2] >= np.power(2, marker_length) or chem_pop[child][2] < 0:
            print "fuck"

#use the rankings to produce mates
def mate_agents(mutatable, order):
    if mutatable == "map":
        parent1 = select_parent(mutatable, np.inf, selecting_scale)
        parent2 = select_parent(mutatable, parent1, selecting_scale)
        if parent1 == parent2:
            print "fuck"
        mate_maps(order[parent1], order[parent2])
    elif mutatable == "chem":
        parent1 = select_parent(mutatable, np.inf, order)
        parent2 = select_parent(mutatable, parent1, order)
        if parent1 == parent2:
            print "fuck"
        mate_chemicals(parent1, parent2)
    else:
        parent1 = select_parent(mutatable, np.inf, order)
        parent2 = select_parent(mutatable, parent1, order)
        if parent1 == parent2:
            print "fuck"
        mate_neurons(parent1, parent2)

#copy child to new location
def copy_agent(mutatable, copy, paste, temp):
    if mutatable == "map":
        if temp == "pop to temp":
            for i in range(map_params):
                temp_map_pop[paste][i] = map_pop[copy][i]
        elif temp == "temp to pop":
            for i in range(map_params):
                map_pop[paste][i] = temp_map_pop[copy][i]
        else:
            for i in range(map_params):
                map_pop[paste][i] = map_pop[copy][i]
    elif mutatable == "chem":
        if temp == "pop to temp":
            for i in range(chem_params):
                temp_chem_pop[paste][i] = chem_pop[copy][i]
        elif temp == "temp to pop":
            for i in range(chem_params):
                chem_pop[paste][i] = temp_chem_pop[copy][i]
        else:
            for i in range(chem_params):
                chem_pop[paste][i] = chem_pop[copy][i]
    else:
        if temp == "pop to temp":
            for i in range(neuron_params):
                temp_neuron_pop[paste][i] = neuron_pop[copy][i]
        elif temp == "temp to pop":
            for i in range(neuron_params):
                neuron_pop[paste][i] = temp_neuron_pop[copy][i]
        else:
            for i in range(neuron_params):
                neuron_pop[paste][i] = neuron_pop[copy][i]

#generate and copy all children
def off_spring(mutatable):
    if mutatable == "map":
        order = rank_fitnesses(mutatable, False)
        for i in range(temp_map_pop_size):
            mate_agents(mutatable, order)
            copy_agent(mutatable, map_pop_size, i, "pop to temp")
        for i in range(temp_map_pop_size):
            copy_agent(mutatable, i, order[map_pop_size-i-1], "temp to pop")
    elif mutatable == "chem":
        order = rank_fitnesses(mutatable, False)
        for i in range(temp_chem_pop_size):
            mate_agents(mutatable, order)
            copy_agent(mutatable, chem_pop_size, i, "pop to temp")
        order = rank_fitnesses(mutatable, True)
        for i in range(temp_chem_pop_size):
            copy_agent(mutatable, i, order[chem_pop_size-i-1], "temp to pop")
    else:
        order = rank_fitnesses(mutatable, False)
        for i in range(temp_neuron_pop_size):
            mate_agents(mutatable, order)
            copy_agent(mutatable, neuron_pop_size, i, "pop to temp")
        order = rank_fitnesses(mutatable, True)
        for i in range(temp_neuron_pop_size):
            copy_agent(mutatable, i, order[neuron_pop_size-i-1], "temp to pop")

for gen in range(number_of_generations):
    for cycle in range(cycles_per_generation):
        for map_agent in range(map_per_cycle):
            fitnesses = []
            fitnesses1 = []
            fitnesses2 = []
            for agent in range(map_pop_size):
                print 'starting agent {}/{}/{}/{}'.format(agent, map_agent, cycle, gen)
                fitness = ball_and_beam_tests(agent, True, False, False)
                # number_of_tests = 7
                # fitness = [np.random.randint(0,3), np.random.randint(3,5), np.random.randint(0,3), np.random.randint(3,5), np.random.randint(0,3), np.random.randint(3,5), np.random.randint(0,3), np.random.randint(3,5)]
                copy_fitness = copy.deepcopy(fitness)
                port_offset += 1
                fitnesses.append(fitness[:])
                fitnesses1.append(fitness)
                fitnesses2.append(copy_fitness)
            off_spring("map")
        for neuron_agent in range(neuron_per_cycle):
            off_spring("neuron")
        for chem_agent in range(chem_per_cycle):
            off_spring("chem")


#Test the population


#evolve on property keeping the others fixed
    #select the best and evolve against it
    #or keep a few of the best and evaluate them in combination

#

print 'DONE!!!'