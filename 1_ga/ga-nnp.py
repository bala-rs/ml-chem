import math
import random
from ase.lattice.cubic import FaceCenteredCubic
from ase.build import surface

"""
Developing Neural Network Potential for nanoparticles with the help of 
genetic algorithm. The NNP will be trained using training data from 
calculations of slab and GA will evolve the set of planes that are
required to construct the NNP for NPs
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

def generate_first_population(population_size):

    population = []

    while len(population) < population_size:

        # Get three random intergers 0-9
        m1 = int(random.random() * 9)
        m2 = int(random.random() * 9)
        m3 = int(random.random() * 9)

        miller_indices = [m1, m2, m3]
        
        # Make sure [0, 0, 0] is not generated!
        if m1 == m2 == m3 == 0:
            print("h, k, l = 0 !!!")
            miller_indices[int(random.random() * 2)] += (int(random.random() * 8) + 1)

        population.append(miller_indices)

    return population


