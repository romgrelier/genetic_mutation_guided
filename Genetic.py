import numpy as np
from typing import List
from random import randint, sample, choices, random
from matplotlib import pyplot


class Individual:
    def __init__(self, genotype: np.array):
        self.genotype: np.array = genotype
        self.fitness = 0
        self.age = 0


def fitness_function(individual: Individual):
    return np.sum(individual.genotype)


def bit_flip(individual: Individual, n=0):
    if n == 0:  # flip the entire genotype
        individual.genotype = np.logical_not(individual.genotype)
    else:  # flip n bits randomly
        ids = sample(range(individual.genotype.size), n)
        individual.genotype[ids] = np.logical_not(individual.genotype[ids])


def random_bit_flip(individual: Individual):
    rate: float = 1.0 / individual.genotype.size

    individual.genotype = np.array([not g if random() < rate else g for g in individual.genotype])


class Genetic:
    def __init__(self, pop_size, genotype_size, mutation_selector):
        self.pop_size = pop_size
        self.genotype_size = genotype_size
        self.population = [Individual(np.zeros(genotype_size, np.bool)) for _ in range(pop_size)]
        self.iter = 0

        for p in self.population:
            p.fitness = fitness_function(p)

        self.mutation_operators_selector = mutation_selector(
            self,
            [
                lambda x: bit_flip(x, 1),
                lambda x: bit_flip(x, 3),
                lambda x: bit_flip(x, 5),
                # lambda x: bit_flip(x, 7),
                # lambda x: random_bit_flip(x),
            ]
        )

    def run(self, iter_max):
        best: np.array = np.zeros(iter_max, np.int)
        worst: np.array = np.zeros(iter_max, np.int)
        mean: np.array = np.zeros(iter_max, np.float)
        p_operator = np.zeros((iter_max, self.mutation_operators_selector.k), np.float)
        quality = np.zeros((iter_max, self.mutation_operators_selector.k), np.float)

        for self.iter in range(iter_max):
            # SELECTION
            parents: List[Individual] = self.selection(copy=True)
            # parents: List[Individual] = self.selection_best()

            # CROSSOVER
            # offspring: List[Individual] = Genetic.multi_point_cross_over(parents, 2)
            offspring: List[Individual] = Genetic.uniform_cross_over(parents)
            # offspring = parents

            # MUTATION
            self.mutation_operators_selector.apply(offspring[0])
            # self.mutation_operators_selector.apply(offspring[1])

            # INSERTION
            self.insertion(offspring)

            # UPDATE
            for p in self.population:
                p.age += 1

            # STATS
            # p_operator[self.iter, :] = self.mutation_operators_selector.p[:]
            # quality[i, :] = self.mutation_operators_selector.q[:]
            mean[self.iter] = sum([p.fitness for p in self.population]) / self.pop_size
            best[self.iter] = max(self.population, key=lambda individual: individual.fitness).fitness
            worst[self.iter] = min(self.population, key=lambda individual: individual.fitness).fitness

        return max(self.population, key=lambda individual: individual.fitness), best, worst, p_operator, mean, quality

    def selection(self, copy=False) -> List[Individual]:
        # random_selection: List[Individual] = sample(self.population, max(int(len(self.population) * 0.05), 5))
        random_selection: List[Individual] = sample(self.population, 5)

        random_selection.sort(key=lambda individual: individual.fitness, reverse=True)

        selection = []
        if not copy:
            selection = [
                random_selection[0],
                random_selection[1]
            ]
        else:
            selection = [
                Individual(np.array(random_selection[0].genotype, copy=True)),
                Individual(np.array(random_selection[1].genotype, copy=True))
            ]

            for s in selection:
                s.fitness = fitness_function(s)

        return selection

    def selection_best(self, n=1, copy=False) -> List[Individual]:
        self.population.sort(key=lambda individual: individual.fitness)

        selection = []
        if not copy:
            selection = [
                self.population[-i] for i in range(0, n)
            ]
        else:
            selection = [
                Individual(np.array(self.population[-i].genotype, copy=True)) for i in range(0, n)
            ]

            for s in selection:
                s.fitness = fitness_function(s)

        return selection

    @staticmethod
    def cross_over(parents: List[Individual]) -> List[Individual]:
        cross_over_point = randint(0, parents[0].genotype.size - 1)

        offspring = [
            Individual(
                np.concatenate((parents[0].genotype[:cross_over_point], parents[1].genotype[cross_over_point:]))),
            Individual(
                np.concatenate((parents[1].genotype[:cross_over_point], parents[0].genotype[cross_over_point:])))
        ]

        for c in offspring:
            c.fitness = fitness_function(c)

        return offspring

    @staticmethod
    def multi_point_cross_over(parents: List[Individual], points: int) -> List[Individual]:
        genotype_size = parents[0].genotype.size

        offspring = [
            Individual(np.zeros(genotype_size, np.bool)),
            Individual(np.zeros(genotype_size, np.bool))
        ]

        begin = 0
        for point in range(points):
            section = np.random.randint(begin, genotype_size)

            if point % 2 == 1:
                offspring[0].genotype[begin:section] = parents[0].genotype[begin:section]
                offspring[1].genotype[begin:section] = parents[1].genotype[begin:section]
            else:
                offspring[0].genotype[begin:section] = parents[1].genotype[begin:section]
                offspring[1].genotype[begin:section] = parents[0].genotype[begin:section]

            begin += (section - begin)

        if points % 2 == 1:
            offspring[0].genotype[begin:] = parents[0].genotype[begin:]
            offspring[1].genotype[begin:] = parents[1].genotype[begin:]
        else:
            offspring[0].genotype[begin:] = parents[1].genotype[begin:]
            offspring[1].genotype[begin:] = parents[0].genotype[begin:]

        for c in offspring:
            c.fitness = fitness_function(c)

        return offspring

    @staticmethod
    def uniform_cross_over(parents: List[Individual]):
        genotype_size = parents[0].genotype.size

        c_1 = [parents[0].genotype[i] if random() > 0.5 else parents[1].genotype[i] for i in range(genotype_size)]
        c_2 = [parents[0].genotype[i] if random() > 0.5 else parents[1].genotype[i] for i in range(genotype_size)]

        offspring = [
            Individual(np.array(c_1, dtype=np.bool)),
            Individual(np.array(c_2, dtype=np.bool))
        ]

        return offspring

    def insertion(self, offspring: List[Individual]):
        for c in offspring:
            self.population.sort(key=lambda individual: individual.fitness)
            self.population[0] = c

    def insertion_remove_older(self, offspring: List[Individual]):
        for c in offspring:
            self.population.sort(key=lambda individual: individual.age)
            self.population[-1] = c


class FixedProbabilities:
    def __init__(self, state, operators):
        self.state = state
        self.operators = operators
        self.k = len(self.operators)
        self.p = np.array([1.0 / self.k for _ in range(self.k)])

    def apply(self, individual):
        operator = np.random.choice(range(self.k), size=1, replace=False, p=self.p)[0]

        # apply the chosen operator
        before_mutation_fitness = individual.fitness
        self.operators[operator](individual)
        individual.fitness = fitness_function(individual)

        improvement = individual.fitness - before_mutation_fitness
        improvement = 0 if improvement < 0 else improvement / individual.genotype.shape[0]

        return improvement


class ProbabilityMatching:
    def __init__(self, state, operators):
        self.state = state
        self.operators = operators
        self.p_min = 0.05
        self.k = len(self.operators)
        self.alpha = 0.9
        self.p = np.array([1.0 / self.k for _ in range(self.k)])
        self.q = np.array([0.0 for _ in range(self.k)], np.float)

    def apply(self, individual: Individual):
        # select operator
        operator = np.random.choice(range(self.k), size=1, replace=False, p=self.p)[0]

        # apply the chosen operator
        before_mutation_fitness: float = individual.fitness
        self.operators[operator](individual)
        individual.fitness = fitness_function(individual)

        # compute the improvement
        # improvement = np.zeros(self.k, np.float)
        improvement = individual.fitness - before_mutation_fitness
        # self.improvements[operator] += 0 if improvement <= 0 else improvement
        improvement = 0 if improvement <= 0 else improvement
        # print(improvement)
        # improvement += 1.0

        # update the quality
        # self.q[operator] = (1 - self.alpha) * self.q[operator] + self.alpha * self.improvements[operator].mean()
        self.q[operator] = (1 - self.alpha) * self.q[operator] + self.alpha * improvement
        # self.q = (1 - self.alpha) * self.q + self.alpha * improvement
        # self.q[operator] += improvement
        # print(self.q)
        # normalization
        # self.q = (self.q - np.min(self.q)) / (np.max(self.q) - np.min(self.q))

        # print(self.q)

        # update the probabilities
        self.p = self.p_min + (1 - self.k * self.p_min) * (self.q / self.q.sum())

        return improvement


class AdaptivePursuit:
    def __init__(self, state, operators):
        self.state = state
        self.operators = operators
        self.p_min = 0.01  # probability min
        self.k = len(self.operators)  # nb operators
        self.p_max = 1 - (self.k - 1) * self.p_min  # probability max
        self.p = np.array([1.0 / self.k for _ in range(self.k)])
        self.q = np.array([1.0 for _ in range(self.k)])
        self.alpha = 0.5
        self.beta = 0.5

    def apply(self, individual: Individual):
        operator = np.random.choice(range(self.k), size=1, replace=False, p=self.p)[0]

        # apply the chosen operator
        before_mutation_fitness = individual.fitness
        self.operators[operator](individual)
        individual.fitness = fitness_function(individual)

        # compute the improvement
        improvement = np.zeros(self.k, np.float)
        improvement[operator] = individual.fitness - before_mutation_fitness
        improvement[operator] = 0 if improvement[operator] <= 0 else improvement[operator]

        # update the quality
        # self.q = (1 - self.alpha) * self.q + self.alpha * improvement
        self.q[operator] = (1 - self.alpha) * self.q[operator] + self.alpha * improvement[operator]
        # self.q[operator] += improvement[operator]

        # normalization
        # self.q = (self.q - np.min(self.q)) / (np.max(self.q) - np.min(self.q))

        # update the probabilities
        i_star = np.argmax(self.q)
        self.p[i_star] = self.p[i_star] + self.beta * (self.p_max - self.p[i_star])
        for i in range(len(self.p)):
            if i != i_star:
                self.p[i] = self.p[i] + self.beta * (self.p_min - self.p[i])

        # print(self.p)

        return improvement


class UCB:
    def __init__(self, state, operators):
        self.state = state
        self.operators = operators
        self.k = len(self.operators)
        self.counter = np.array([0 for _ in range(self.k)])
        self.q = np.array([0.0 for _ in range(self.k)])
        self.p = np.array([1.0 / self.k for _ in range(self.k)])
        self.c = 1.0  # exploration coefficient

    def apply(self, individual: Individual):
        # compute probabilities
        self.p = self.q + self.c * np.sqrt((2 * np.log(self.counter.sum())) / self.counter)
        # self.p /= np.sum(self.p)

        # choose the operator
        operator = np.argmax(self.p)

        # apply the chosen operator
        before_mutation_fitness = individual.fitness
        self.operators[operator](individual)
        individual.fitness = fitness_function(individual)

        # compute the improvement
        improvement = individual.fitness - before_mutation_fitness
        improvement = 0 if improvement <= 0 else improvement

        self.counter[operator] += 1
        # print(improvement)
        # update the quality
        self.q[operator] = (1 - (1 / self.counter[operator])) * self.q[operator] + (1 / self.counter[operator]) * improvement

        # normalization
        # print(self.q)
        # self.q = (self.q - np.min(self.q)) / (np.max(self.q) - np.min(self.q))
        # print((np.max(self.q) - np.min(self.q)))
        # print(self.q)

        # input()

        # return improvement


class QLearning:
    def __init__(self, state, operators):
        self.state = state
        self.operators = operators
        self.k = len(self.operators)
        self.alpha = 1.0  # learning rate
        self.discount = 1.0  # discount factor
        # self.q = np.random.random((501, self.k))  # quality matrix
        self.q = np.load('q_matrix.model.npy')

    def save_model(self):
        # save the q matrix
        np.save("q_matrix.model", self.q)

    def apply(self, individual: Individual, offline=False):
        state = individual.fitness
        operator = np.argmax(self.q[state, :])

        # apply the chosen operator
        before_mutation_fitness = individual.fitness
        self.operators[operator](individual)
        individual.fitness = fitness_function(individual)

        # compute the improvement
        improvement = individual.fitness - before_mutation_fitness
        improvement = 0 if improvement < 0 else improvement  # / individual.genotype.shape[0]

        if not offline:
            state = before_mutation_fitness

            # update the q matrix
            self.q[state, operator] = (1 - self.alpha) + self.q[state, operator] + self.alpha * (
                        improvement + self.discount * np.max(self.q[state, :]))

            self.save_model()

        return improvement


# best = np.zeros((20000, 5))
# run = 5
#
# i = 0
# for p in [10, 20, 50, 100, 200]:
#     best_local = np.zeros((20000, run), np.float)
#     for r in range(run):
#         g = Genetic(pop_size=p, genotype_size=500, mutation_selector=AdaptivePursuit)
#
#         result, best_history, worst_history, p_history, mean_history, quality = g.run(20000)
#
#         best_local[:, r] = best_history[:]
#
#     best[:, i] = best_local.mean(axis=1)
#     i += 1
#
# i = 0
# for p in [10, 20, 50, 100, 200]:
#     pyplot.plot(best[:, i], label=p, linewidth=1)
#     i += 1
#
# pyplot.legend()
# pyplot.show()


selectors = {
    "FixedProbabilities": FixedProbabilities,
    "ProbabilityMatching": ProbabilityMatching,
    # "AdaptivePursuit": AdaptivePursuit,
    # "UCB": UCB,
    # "QLearning": QLearning
}

linewidth = 1
run = 5
max_iter = 20000

best = np.zeros((max_iter, len(selectors)))

i = 0
for k, selector in selectors.items():
    best_for_selector = np.zeros((max_iter, run), np.float)

    for r in range(run):
        g = Genetic(pop_size=20, genotype_size=500, mutation_selector=selector)

        result, best_history, worst_history, p_history, mean_history, quality = g.run(max_iter)

        best_for_selector[:, r] = best_history[:]

        # pyplot.plot(best_history, label="best", linewidth=linewidth)
        # pyplot.plot(worst_history, label="worst", linewidth=linewidth)
        # pyplot.plot(mean_history, label="mean", linewidth=linewidth)
        # # pyplot.plot(imps, label="improvements", linewidth=linewidth)
        # pyplot.legend()
        # pyplot.show()
        #
        # pyplot.plot(p_history[:, 0], label="1 flip", linewidth=linewidth)
        # pyplot.plot(p_history[:, 1], label="3 flip", linewidth=linewidth)
        # pyplot.plot(p_history[:, 2], label="5 flip", linewidth=linewidth)
        # # pyplot.plot(p_history[:, 3], label="7 flip", linewidth=linewidth)
        # pyplot.legend()
        # pyplot.show()

    best[:, i] = best_for_selector.mean(axis=1)
    print(f"{k} (mean) : {best[-1, i]}")
    i += 1

i = 0
for k, selector in selectors.items():
    pyplot.plot(best[:, i], label=k, linewidth=linewidth)
    i += 1

pyplot.legend()
pyplot.show()
