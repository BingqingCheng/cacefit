import sys
sys.path.append('../../cace/')
import os
import pickle
import numpy as np
import torch
import torch.nn as nn

import cace
from cace.representations.cace_representation import Cace
from cace.calculators import CACECalculator

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write

init_conf = read('../../datasets/md17_ethanol/test-n1000.xyz', '0')
cace_nnp = torch.load(sys.argv[1], map_location=torch.device('cpu'))

# load the avge0 dict from a file if possible
if os.path.exists('avge0.pkl'):
    with open('avge0.pkl', 'rb') as f:
        avge0 = pickle.load(f)
else:
    xyz = read('../../datasets/md17_ethanol/train-n1000.xyz',':')
    avge0 = cace.tools.compute_average_E0s(xyz)
    # save the avge0 dict to a file
    with open('avge0.pkl', 'wb') as f:
        pickle.dump(avge0, f)

calculator = CACECalculator(model_path=cace_nnp, 
                            device='cpu', 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            energy_units_to_eV=0.0433641153087705,
                            compute_stress=False,
                           atomic_energies=avge0)

init_conf.set_calculator(calculator)

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

temperature = float(sys.argv[2])# in K

# Set initial velocities using Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(init_conf, temperature * units.kB)


def print_energy(a):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.4feV  Ekin = %.4feV (T=%3.0fK)  '
          'Etot = %.4feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

def write_frame():
        dyn.atoms.write('md_ethanol-T-'+str(temperature)+'.xyz', append=True)

dyn = Langevin(init_conf, 1*units.fs, temperature_K=temperature, friction=1e-3)

# Define the NPT ensemble
#NPTdamping_timescale = 10 * units.fs  # Time constant for NPT dynamics
#NVTdamping_timescale = 100 * units.fs  # Time constant for NVT dynamics (NPT includes both)
#dyn = NPT(init_conf, timestep=1 * units.fs, temperature_K=temperature,
#          ttime=NVTdamping_timescale, pfactor=None, #0.1*NPTdamping_timescale**2,
#          externalstress=0.0)

#dyn = NPTBerendsen(init_conf, timestep=1 * units.fs, temperature_K=temperature,
#                   taut=NVTdamping_timescale, pressure_au=0.0,
#                   taup=NPTdamping_timescale, compressibility_au=1.)

dyn.attach(write_frame, interval=1000)

dyn.attach(MDLogger(dyn, init_conf, 'nvt-T-'+str(temperature)+'.log', header=True, stress=False,
           peratom=False, mode="w"), interval=20)

# Run the MD simulation
n_steps = 30000
for step in range(n_steps):
    print_energy(a=init_conf)
    dyn.run(10)
