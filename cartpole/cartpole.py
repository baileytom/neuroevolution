import gym
import neat
import numpy as np

def eval_genomes(genomes, config):
    env = gym.make('CartPole-v0')
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(net.activate(state))
            state, reward, done, info = env.step(action)
            genome.fitness += reward

def render_for_net(net):
    env = gym.make('CartPole-v0')
    state = env.reset()
    while True:
        env.render()
        action = np.argmax(net.activate(state))
        state, _, _, _ = env.step(action)
            
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-cartpole')

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(False))
winner = p.run(eval_genomes)

render_for_net(neat.nn.FeedForwardNetwork.create(winner, config))
