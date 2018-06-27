import math
import random
import operator
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork

from ase import Atoms
from ase.calculators.emt import EMT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.build import surface
from ase.build import make_supercell
from ase.db import connect

"""
Developing Neural Network Potential for nanoparticles with the help of 
genetic algorithm. The NNP will be trained using training data from 
calculations of slab and GA will evolve the set of planes that are
required to construct the NNP for NPs

GA tutorial: https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9

"""

def fitness(reference_energy, nnp_energy):
    """
    :param reference_energy: reference energy calculated by EAM
    :nnp_energy: predicted energy by NNP
    :return: squared error
    """

    return math.pow(reference_energy - nnp_energy, 2)


def generate_slab(miller_indices, layers):

    lattice = FaceCenteredCubic('Cu')

    return surface(lattice, miller_indices, layers, vacuum=15)

def generate_first_population(population_size, mi_per_individual=10):

    """
    The target is to find the best set of miller_indices that can represent the 
    chemical environment of atoms in NP. So "an individual" of the population
    will be composed of several (fixed to 10 for now) miller_indices.
    """

    population = []

    while len(population) < population_size:

        individual = []
        while len(individual) < mi_per_individual:
            # Get three random intergers 0-9
            m1 = int(random.random() * 9)
            m2 = int(random.random() * 9)
            m3 = int(random.random() * 9)

            miller_indices = [m1, m2, m3]
        
            # Make sure [0, 0, 0] is not generated!
            if m1 == m2 == m3 == 0:
                print("h, k, l = 0 !!!")
                miller_indices[int(random.random() * 2)] += (int(random.random() * 8) + 1)

            individual.append(miller_indices)
        population.append(individual)

    return population

def generate_training_set(individual, db_name):

    """
    Do MD using EMT calculator with each slab for 10 steps and add it to train.db
    """

    db = connect(db_name)
    for miller_indices in individual:

        # Do MD
        slab = generate_slab(miller_indices, 5)
        slab = make_supercell(slab, P=[[3, 1, 1], [1, 3, 1], [1, 1, 1]])

        slab.set_calculator(EMT())
        slab.get_potential_energy()

        db.write(slab)

        MaxwellBoltzmannDistribution(slab, 300. * units.kB)
        dyn = VelocityVerlet(slab, dt=1. * units.fs)
        for i in range(1, 10):
            dyn.run(10)
            db.write(slab)


def compute_population_performance(population, reference):

    for individual in population:

        # Construct NNP
        nnp = Amp(descriptor=Gaussian(),
                  model=NeuralNetwork(hiddenlayers=(5, 5)),
                  cores=8)

        # Predict energy of NP
