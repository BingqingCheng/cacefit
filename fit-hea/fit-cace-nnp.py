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

cutoff = 4.5
collection = cace.tasks.get_dataset_from_xyz(train_path='n-24630.xyz',
                                 valid_fraction=0.05,
                                 cutoff=cutoff
                                 )

batch_size = 10

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=20,
                              )

use_device = 'cuda'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")


logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 44, 45, 46, 47, 71, 72, 73, 74, 77, 78, 79],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=8,
    max_l=3,
    max_nu=3,
    num_message_passing=1,
    type_message_passing=["Bchi"],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    device=device,
    timeit=False
           )

cace_representation.to(device)
logging.info(f"Representation: {cace_representation}")

atomwise = cace.modules.Atomwise(n_layers=3,
                                 output_key='CACE_energy',
                                 n_hidden=[24,12],
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
    loss_weight=0.1
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

logging.info("creating training task")

optimizer_args = {'lr': 1e-2}  
scheduler_args = {'mode': 'min', 'factor': 0.8, 'patience': 10}

for i in range(8):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        #scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=True,
        ema_start=10,
        warmup_steps=5,
    )

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=25, screen_nan=False)

task.save_model('hea-model.pth')
cace_nnp.to(device)

logging.info(f"Second train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1
)

task.update_loss([energy_loss, force_loss])
logging.info("training")
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)


task.save_model('hea-model-2.pth')
cace_nnp.to(device)

logging.info(f"Third train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10 
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)

task.save_model('hea-model-3.pth')
cace_nnp.to(device)



logging.info(f"Fourth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=100
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)

task.save_model('hea-model-4.pth')
cace_nnp.to(device)

logging.info(f"Fifth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)

task.save_model('hea-model-5.pth')
