import spynnaker7.pyNN as p
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spinn_front_end_common.utilities.globals_variables import get_simulator

import pylab
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
import sys, os
import time
import socket
import numpy as np
import math
import csv

def test_agent(weight):
    # Setup pyNN simulation
    weight = weight / 100.
    weight = weight * -1
    print "da weight = ", weight
    p.setup(timestep=1.0)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)

    receive_pop_size = 1
    hidden_pop_size = 1
    output_size = 1

    # Create input population and connect break out to it
    receive_on_pop = p.Population(receive_pop_size, p.IF_cond_exp, {}, label="receive_pop")

    # Create output population and remaining population
    output_pop = p.Population(output_size, p.IF_cond_exp, {}, label="output_pop")

    hidden_node_pop = p.Population(hidden_pop_size, p.IF_cond_exp, {}, label="hidden_pop")
    hidden_node_pop.record()
    receive_on_pop.record()
    output_pop.record()

    spikes_in = p.Population(1, p.SpikeSourceArray, {'spike_times': [100]}, label='spike')

    p.Projection(output_pop, receive_on_pop, p.AllToAllConnector(weights=weight))
    p.Projection(receive_on_pop, hidden_node_pop, p.AllToAllConnector(weights=weight))
    p.Projection(hidden_node_pop, output_pop, p.AllToAllConnector(weights=weight))
    p.Projection(spikes_in, receive_on_pop, p.AllToAllConnector(weights=weight))

    runtime = 1000
    p.run(runtime)

    pylab.figure()
    spikes_on = receive_on_pop.getSpikes()
    ax = pylab.subplot(1, 3, 1)#4, 1)
    pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    pylab.xlabel("Time (ms)")
    pylab.ylabel("neuron ID")
    pylab.axis([0, runtime, -1, receive_pop_size + 1])
    # pylab.show()
    # pylab.figure()
    spikes_on = hidden_node_pop.getSpikes()
    ax = pylab.subplot(1, 3, 2)#4, 1)
    pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    pylab.xlabel("Time (ms)")
    pylab.ylabel("neuron ID")
    pylab.axis([0, runtime, -1, hidden_pop_size + 1])
    # pylab.show()
    # pylab.figure()
    spikes_on = output_pop.getSpikes()
    ax = pylab.subplot(1, 3, 3)#4, 1)
    pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    pylab.xlabel("Time (ms)")
    pylab.ylabel("neuron ID")
    pylab.axis([0, runtime, -1, output_size + 1])
    pylab.show()


    # End simulation
    p.end()

for i in range(8,10,1):
    test_agent(i)