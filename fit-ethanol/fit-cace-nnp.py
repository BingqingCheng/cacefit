#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../../cace/')

import numpy as np
import torch
import torch.nn as nn
import logging

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')

import ase
xyz = ase.io.read('../../datasets/md17_ethanol/train-n1000.xyz', ':')
avge0 = cace.tools.compute_average_E0s(xyz)

collection = cace.tasks.get_dataset_from_xyz(train_path='../../datasets/md17_ethanol/train-n1000.xyz',
                                 valid_fraction=0.1,
                                 energy_key='energy',
                                 forces_key='forces',
                                 atomic_energies=avge0)

cutoff = 4
batch_size = 10

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              cutoff=cutoff)

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=100,
                              cutoff=cutoff)

use_device = 'cuda'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")


logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff, p=5)

cace_representation = Cace(
    zs=[1, 6, 8],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=4,
    max_nu=3,
    device=device,
    num_message_passing=1,
    type_message_passing=["M", "Ar", "Bchi"],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    timeit=False
           )

cace_representation.to(device)
logging.info(f"Representation: {cace_representation}")

atomwise = cace.modules.Atomwise(n_layers=3,
                                 output_key='CACE_energy',
                                 n_hidden=[32,16],
                                 residual=False,
                                 use_batchnorm=False,
                                 add_linear_nn = True)

forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

logging.info("building CACE NNP")
cace_nnp = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)

cace_nnp.to(device)


logging.info(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

energy_loss_2 = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e',
    per_atom=False
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2}  
scheduler_args = {'mode': 'min', 'factor': 0.8, 'patience': 10}

for _ in range(9):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=True,
        ema_start=10,
        warmup_steps=10,
    )

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=300, screen_nan=False)

task.fit(train_loader, valid_loader, epochs=300, screen_nan=False)
task.save_model('eth-model-1.pth')

task.update_loss([energy_loss_2, force_loss])

task.fit(train_loader, valid_loader, epochs=1000, screen_nan=False)
task.save_model('eth-model-2.pth')
logging.info("Finished!")
