import spynnaker7.pyNN as p
import time
import copy
import pylab
import numpy as np
import threading
from threading import Condition
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pyNN.random import RandomDistribution# as rand
#import spynnaker8.spynakker_plotting as splot
from pympler.tracker import SummaryTracker
import spinn_breakout

input_size = 160*128
hidden_size = 100
output = 100

p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)

poisson_rate = {'rate': 15}
tracker = SummaryTracker()
tracker.print_diff()

input_pop = p.Population(input_size, p.IF_cond_exp, {}, label='input')
hidden_pop = p.Population(hidden_size, p.IF_cond_exp, {},  label='hidden')
output_pop = p.Population(output, p.IF_cond_exp, {},  label='output')
breakout = p.Population(1, spinn_breakout.Breakout, {}, label="breakout")
print "after populations"
tracker.print_diff()
a = p.Projection(input_pop, hidden_pop, p.AllToAllConnector(weights=0.5))
print "after i->h"
tracker.print_diff()
a = p.Projection(breakout, hidden_pop, p.AllToAllConnector(weights=0.5))
print "after b->h"
tracker.print_diff()
b = p.Projection(input_pop, output_pop, p.AllToAllConnector(weights=2))
print "after i->o"
tracker.print_diff()
c = p.Projection(hidden_pop, output_pop, p.AllToAllConnector(weights=2))
print "after h->o"
tracker.print_diff()
d = p.Projection(input_pop, input_pop, p.AllToAllConnector(weights=2))
print "after i->i"
tracker.print_diff()
a = p.Projection(input_pop, hidden_pop, p.AllToAllConnector(weights=2))
print "after i->h 2"
tracker.print_diff()

print "finished"

