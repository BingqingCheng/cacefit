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

water_conf = read('liquid-64.xyz', '0')
init_conf = water_conf.repeat((2,2,2))
cace_nnp = torch.load(sys.argv[1], map_location=torch.device('cpu'))

preprocessor = cace.modules.Preprocess()
cace_nnp.input_modules = nn.ModuleList([preprocessor])
cace_nnp.output_modules[1].calc_stress = True
cace_nnp.model_outputs.append('stress')

# load the avge0 dict from a file if possible
if os.path.exists('avge0.pkl'):
    with open('avge0.pkl', 'rb') as f:
        avge0 = pickle.load(f)
else:
    xyz = read('../../datasets/water/water.xyz',':')
    avge0 = cace.tools.compute_average_E0s(xyz)
    # save the avge0 dict to a file
    with open('avge0.pkl', 'wb') as f:
        pickle.dump(avge0, f)

calculator = CACECalculator(model_path=cace_nnp, 
                            device='cpu', 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            compute_stress=True,
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
        dyn.atoms.write('md_water-T-'+str(temperature)+'.xyz', append=True)

# Define the NPT ensemble
NPTdamping_timescale = 10 * units.fs  # Time constant for NPT dynamics
NVTdamping_timescale = 100 * units.fs  # Time constant for NVT dynamics (NPT includes both)
dyn = NPT(init_conf, timestep=1 * units.fs, temperature_K=temperature,
          ttime=NVTdamping_timescale, pfactor=None, #0.1*NPTdamping_timescale**2,
          externalstress=0.0)

dyn.attach(write_frame, interval=100)

dyn.attach(MDLogger(dyn, init_conf, 'npt-T-'+str(temperature)+'.log', header=True, stress=True,
           peratom=False, mode="w"), interval=10)

# Run the MD simulation
n_steps = 1000
for step in range(n_steps):
    print_energy(a=init_conf)
    dyn.run(1000)
