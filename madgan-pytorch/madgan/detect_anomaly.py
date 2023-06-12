import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import preprocessing
import models
import constants
import anomaly

def get_prediction(data_path:str):

    input_data = preprocessing.read_data(data_path)
    print(input_data.shape)
    input_data = preprocessing.reshape(512, input_data)
    print(input_data.shape)
    input_data = preprocessing.normalize(input_data)
    print(input_data.shape)
    input_data = input_data[:-1 * (input_data.shape[0] % constants.LATENT_SPACE_DIM)].reshape(input_data.shape[0] // constants.LATENT_SPACE_DIM, 512, constants.LATENT_SPACE_DIM)

    input_data = input_data[len(input_data) // 2:len(input_data) // 2 + 30, :]

    print(input_data.shape)

    generator = models.Generator(latent_space_dim=constants.LATENT_SPACE_DIM, hidden_units=100, output_dim=input_data.shape[-1])
    discriminator = models.Discriminator(input_dim=input_data.shape[-1], hidden_units=100, n_lstm_layers=2, add_batch_mean=False)

    generator.load_state_dict(torch.load('../models/generator_500.pt')['weights'])
    discriminator.load_state_dict(torch.load('../models/discriminator_500.pt')['weights'])

    detector = anomaly.AnomalyDetector(discriminator=discriminator, generator=generator, latent_space_dim=constants.LATENT_SPACE_DIM)

    input_tensor = torch.from_numpy(input_data)
    reconstruction = detector._generate_reconstruction(input_tensor)

    anomaly_score = detector.predict_proba(input_tensor)

    return input_data, reconstruction, anomaly_score

def plot_result(recon, origin):
    plt.figure(figsize=[16, 3])
    plt.plot(recon.reshape(recon.shape[0] * recon.shape[1] * recon.shape[2])[:])
    # plt.figure(figsize=[16, 3])
    plt.plot(origin.reshape(origin.shape[0] * origin.shape[1] * origin.shape[2])[:])
    plt.show()


origin, recon, anomaly_score = get_prediction('../data/data.csv')

print(origin.shape, recon.shape, anomaly_score.shape)

plot_result(recon.numpy(), origin)