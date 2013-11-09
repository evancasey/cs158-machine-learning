from learning import *
import pdb
import math
import random
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def testNeuralNetLearner(data, rounds):

		# change stuff
		if data == "restaurant":
		    attributeNames = "Alternate, Bar, Fri/Sat, Hungry, Patrons, Price, Raining, Reservation, Type, WaitEstimate, WillWait"
		    ds = DataSet(name='../data/restaurant', attrnames=attributeNames)
		elif data == "cancer":
		    attributeNames = "Class, age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat"
		    ds = DataSet(name='../breast-cancer/breast-cancer', attrnames=attributeNames)
		elif data == "new_bands":
		    attributeNames = "timestamp, cylinder_number, customer, job_number, paper_type, ink_type, paper_mill_location, plating_tank, proof_cut, viscosity, caliper, ink_temperature, humifity, roughness, blade_pressure, varnish_pct, press_speed, ink_pct, band_type"
		    ds = DataSet(name='../data/trimmed_bands', attrnames=attributeNames)


if __name__ == "__main__":

		# some stuff