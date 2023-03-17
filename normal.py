import numpy as np
import random
from scipy.stats import norm
from scipy.stats import gamma
import pickle
import matplotlib.pyplot as plt
import scipy


NUM_PARTICLES = 100
PARTICLE_VARIANCE = 0.3
TIMESTEP = 0.01
PROPAGATE_VARIANCE = 0.5
HORIZON = 10

# Policy
TIME_UNTIL_COLLISION = 3  # number of seconds before a collision
THRESHOLD = 0.80  # min percent chance of collision that leads to some action

with open("ttcs_history.pkl", "rb") as f:
    ttcs_history = pickle.load(f)


class Particle:
    def __init__(self, mean, variance, weight):
        self.mean = mean
        self.variance = variance
        self.weight = weight


def init_particles():  # init particles with a uniform dist Uni(0, 10)
    particles = []
    for i in range(NUM_PARTICLES):
        init_mean = random.uniform(0, HORIZON)
        particles.append(Particle(init_mean, PARTICLE_VARIANCE, 1/NUM_PARTICLES))
    return particles


def get_measurement(arr):
    return np.min([np.min(arr), HORIZON])


def set_weights(measurement, particles):
    weights = []
    for particle in particles:  # get weight = ln(P(measurement | distribution of particle i))
        weight = norm.logpdf(measurement, particle.mean, particle.variance)  # GAMMA?
        weights.append(weight)
    normalized_weights = scipy.special.softmax(weights)  # scipy's built-in function takes care of the over/undeflow issues
    for i in range(NUM_PARTICLES):
        particles[i].weight = normalized_weights[i]  # assign each weight to the normalized version
    return particles


def get_dist(particles):
    PDF = np.zeros(1001)
    t = np.linspace(0, HORIZON, 1001)
    for particle in particles:  # weighted sum of all the particle pdfs
        dist = norm(particle.mean, particle.variance)   # GAMMA?
        particle_pdf = dist.pdf(t)
        PDF += particle.weight * particle_pdf
    return t, PDF


def resample(particles):
    weights = [particle.weight for particle in particles]
    particle_indices = np.random.choice(NUM_PARTICLES, size=NUM_PARTICLES, p=weights)
    new_particles = []
    for i in particle_indices:
        mean = particles[i].mean    # GAMMA?
        new_particle = Particle(np.random.normal(mean, PARTICLE_VARIANCE), PARTICLE_VARIANCE, None) # weight should be 1/NUM_PARTICLES?  # GAMMA?
        new_particles.append(new_particle)
    return new_particles


def propagate(particles):
    for particle in particles:
        particle.mean = particle.mean - TIMESTEP + PROPAGATE_VARIANCE * np.random.randn()    # GAMMA?
    return particles


def get_policy(PDF, t, ttc, thresh):
    # makes assumption that time to collision (ttc) is inside of t (timesteps)
    lb, ub = 0, ttc  # define lower and upper bounds
    lb_index = np.argmax(t >= lb)  # find index of lower bound timestep in t (right now this us just 0)
    ub_index = np.argmax(t >= ub)  # find index of upper bound timestep in t
    pdf_interval = PDF[lb_index:ub_index]
    t_interval = t[lb_index:ub_index]
    area = np.trapz(pdf_interval, t_interval)  # get area under the curve of the pdf for these two intervals
    if area <= thresh:
        return "No collision predicted"
    else:
        return "Collision predicted"


def main():
    policies = []
    particles = init_particles()
    for i in range(len(ttcs_history)):
        print(i+1, get_measurement(ttcs_history[i]))
        measurement = get_measurement(ttcs_history[i])
        particles = set_weights(measurement, particles)
        t, PDF = get_dist(particles)
        p = get_policy(PDF, t, TIME_UNTIL_COLLISION, THRESHOLD)
        policies.append(p)
        particles = resample(particles)
        particles = propagate(particles)
    print(policies)
    return policies


if __name__ == '__main__':
    main()
