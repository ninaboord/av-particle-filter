import numpy as np
import random
import pickle
import scipy
import torch

NUM_PARTICLES = 100
PARTICLE_VARIANCE = 0.3
TIMESTEP = 0.01
PROPAGATE_VARIANCE = 0.5
HORIZON = 10

with open("ttcs_history.pkl", "rb") as f:
    ttcs_history = pickle.load(f)


class Particle:
    def __init__(self, mode, variance, weight):
        self.mode = mode
        self.variance = variance
        self.alpha = (self.mode ** 2 + 2 * self.variance + self.mode * np.sqrt(
            self.mode ** 2 + 4 * self.variance)) / (2 * self.variance)
        self.beta = (self.mode + np.sqrt(self.mode ** 2 + 4 * self.variance)) / (2 * self.variance)
        self.weight = weight
        self.distribution_type = torch.distributions.gamma.Gamma

    def update_mode(self, new_mode):
        self.mode = new_mode
        self.alpha = (self.mode ** 2 + 2 * self.variance + self.mode * np.sqrt(
            self.mode ** 2 + 4 * self.variance)) / (2 * self.variance)
        self.beta = (self.mode + np.sqrt(self.mode ** 2 + 4 * self.variance)) / (2 * self.variance)


def init_particles():  # init particles with a uniform dist Uni(0, 10)
    particles = []
    for i in range(NUM_PARTICLES):
        init_mode = random.uniform(0, HORIZON)
        particles.append(Particle(init_mode, PARTICLE_VARIANCE, 1 / NUM_PARTICLES))
    return particles


def get_measurement(arr):
    return np.min([np.min(arr), HORIZON])


def set_weights(measurement, particles):
    eps = 1e-08
    if measurement == 0:
        measurement += eps
    weights = []
    for particle in particles:  # get weight = ln(P(measurement | distribution of particle i))
        dist = particle.distribution_type(particle.alpha, particle.beta)
        weight = dist.log_prob(measurement)
        weights.append(weight.item())
    normalized_weights = torch.nn.functional.softmax(torch.Tensor(weights))  # use torch softmax function
    for i in range(NUM_PARTICLES):
        particles[i].weight = normalized_weights[i].item()  # assign each weight to the normalized version
    return particles


def get_dist(particles):
    PDF = torch.zeros(10001)
    t = torch.linspace(0.0001, HORIZON, 10001)
    for particle in particles:  # weighted sum of all the particle pdfs
        dist = particle.distribution_type(particle.alpha, particle.beta)
        particle_pdf = torch.exp(dist.log_prob(t))
        PDF += particle.weight * particle_pdf
    return t, PDF


def resample(particles):
    weights = [particle.weight for particle in particles]
    sum_weights = sum(weights)
    weights = [w / sum_weights for w in weights]
    particle_indices = np.random.choice(NUM_PARTICLES, size=NUM_PARTICLES, p=weights)
    new_particles = []
    for i in particle_indices:
        sampled_mode = particles[i].distribution_type(particles[i].alpha, particles[i].beta).sample().item()
        new_particle = Particle(sampled_mode, PARTICLE_VARIANCE, None)
        new_particles.append(new_particle)
    return new_particles


def propagate(particles):
    for particle in particles:
        new_mode = particle.mode - TIMESTEP + PROPAGATE_VARIANCE * np.random.randn()
        particle.update_mode(new_mode)
    return particles


def get_policy(particles):
    t_threshold = 3
    p_threshold = 0.8
    t, pdf = get_dist(particles)
    idx_t_threshold = int(t_threshold / HORIZON * t.shape[0])
    p_collision = scipy.integrate.simpson(pdf[:idx_t_threshold], x=t[:idx_t_threshold])
    p_no_collision = 1 - p_collision
    handover = True if p_collision >= p_threshold else False
    return p_collision, p_no_collision, handover


def main():
    policies = []
    particles = init_particles()
    for i in range(len(ttcs_history)):
        print(i + 1, get_measurement(ttcs_history[i]))
        measurement = get_measurement(ttcs_history[i])
        particles = set_weights(measurement, particles)
        p = get_policy(particles)
        policies.append(p)
        particles = resample(particles)
        particles = propagate(particles)
    print(policies)
    return policies


if __name__ == '__main__':
    main()
