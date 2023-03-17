import numpy as np
import random
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
import scipy


NUM_PARTICLES = 100
PARTICLE_VARIANCE = 0.2
TIMESTEP = 0.01
PROPAGATE_VARIANCE = 0.5
HORIZON = 10

# Policy
TIME_UNTIL_COLLISION = 5  # number of seconds before a collision
THRESHOLD = 0.01  # min percent chance of collision that leads to some action

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


# def softmax(arr):
#     a = np.min(arr)
#     exp_arr = np.exp(arr+ a)
#     exp_sum = np.sum(exp_arr)
#     return exp_arr / exp_sum


def set_weights(measurement, particles):
    weights = []
    for particle in particles:  # get weight = ln(P(measurement | distribution of particle i))
        weight = norm.logpdf(measurement, particle.mean, particle.variance)
        # if weight:  # if/else prevents math.log(0) error. setting weight to 0 if norm.pdf is about equal to 0
        #     weight = math.log(weight)
        # else:
        #     weight = 0
        weights.append(weight)  
    normalized_weights = scipy.special.softmax(weights) #scipy's built-in function takes care of the over/undeflow issues
    for i in range(NUM_PARTICLES):
        particles[i].weight = normalized_weights[i]  # assign each weight to the normalized version
    # print("nmlzed weights", normalized_weights)
    return particles


def get_dist(particles):
    PDF = np.zeros(1001)
    t = np.linspace(0, HORIZON, 1001)
    for particle in particles:  # weighted sum of all the particle pdfs
        dist = norm(particle.mean, particle.variance)
        particle_pdf = dist.pdf(t)
        PDF += particle.weight * particle_pdf
    return t,PDF


def resample(particles):
    weights = [particle.weight for particle in particles]
    particle_indices = np.random.choice(NUM_PARTICLES, size=NUM_PARTICLES, p=weights)
    new_particles = []
    for i in particle_indices:
        mean = particles[i].mean
        new_particle = Particle(np.random.normal(mean, PARTICLE_VARIANCE), PARTICLE_VARIANCE, None)
        new_particles.append(new_particle)
    return new_particles


def propagate(particles):
    for particle in particles:
        particle.mean = particle.mean - TIMESTEP + PROPAGATE_VARIANCE * np.random.randn()
    return particles

def save_dist(particles,obs,i):
    plt.rc('text', usetex=True)
    t, pdf = get_dist(particles)
    particles_x = [p.mean for p in particles]
    obs = np.minimum(obs,HORIZON)
    plt.clf()
    plt.plot(t,pdf)
    plt.scatter(particles_x,0.1 * np.ones_like(particles_x),s=1)
    plt.scatter(obs,0.1 * np.ones_like(obs))
    plt.xlim(0,10)
    plt.ylim(0,2)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$p(collision)$")
    plt.tight_layout()
    plt.savefig(f"./test/{i:03d}.png",dpi=600)




def main():
    policies = []
    particles = init_particles()
    for i in range(len(ttcs_history)):
        print(i+1, get_measurement(ttcs_history[i]))
        measurement = get_measurement(ttcs_history[i])
        particles = set_weights(measurement, particles)
        # PDF = get_dist(particles)  # should be able to extract policy from this
        #save_dist(particles,ttcs_history[i],i+1)
        particles = resample(particles)
        particles = propagate(particles)
    # print(policies)
    return policies

    # print statements for testing
    # for particle in particles:
    #     print("mean", particle.mean)
    #     print("var", particle.variance)
    #     print("w", particle.weight)
    # print("PDF", PDF)


if __name__ == '__main__':
    main()
