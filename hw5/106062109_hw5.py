import numpy as np
import sourcedefender
from HomeworkFramework import Function
import math

class CMAES_optimizer(Function):
    def __init__(self, target_func):
        super().__init__(target_func)
        # number of objective variables/problem dimension
        self.nn = self.f.dimension(target_func) 
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
         
        self.target_func = target_func
        self.eval_times = 0 # equivalent to g , g is number of generations
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.nn)

        #strategy parameter----------------
        self.temp = np.random.random(self.nn) 
        # objective variables initial point
        self.xmean = np.copy(self.temp)
        # coordinate wise standard deviation (step-size)
        self.sigma = 0.25
        # population size, offspring number
        self.Lambda = 5 + int(3 * np.log(self.nn))
        self.mu = int(self.Lambda / 2)
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        # normalize recombination weights array
        self.weights = np.array([w / sum(self.weights) for w in self.weights])
        # variance-effective size of mu

        self.mueff = 1 / np.sum(np.power(w, 2) for w in self.weights)
        #strategy parameter----------------------------------
        # time constant for cumulation for C
        self.cc = (4 + self.mueff / self.nn) / (self.nn + 3 + 2 * self.mueff / self.nn)
        # t-const for cumulation for sigma control
        self.cs = (self.mueff + 6) / (self.nn + self.mueff + 4)
        # learning rate for rank-one update of C
        self.c1 = 2 / ((self.nn + 1.3) ** 2 + self.mueff)
        # for rank-mu update
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.nn + 2) ** 2 + self.mueff)])
        # damping for sigma
        self.damps = 1 + self.cs + 2 * max([0, ((self.mueff - 1) / self.nn) ** 0.5 - 1])
        # init dynamic (internal) strategy parameters and constants------------------------------
        # evolution paths for C and sigma
        self.pc, self.ps = np.zeros(self.nn), np.zeros(self.nn)
        # B defines the coordinate system, diagonal matrix D defines the scaling covariance matrix.
        self.B, self.C, self.M = np.eye(self.nn), np.eye(self.nn), np.eye(self.nn)
        self.D = np.ones(self.nn)

    def step(self):
        # Sample
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = self.D ** 0.5
        fitness = np.zeros((self.Lambda, self.nn))
        arz = np.zeros((self.Lambda, self.nn))
        arx = np.zeros((self.Lambda, self.nn))

        fitvals = np.zeros(self.Lambda)
        for i in range(self.Lambda):
            arz[i] = np.random.normal(0, 1, self.nn)
            arx[i] = np.dot(self.B * self.D, arz[i])
            fitness[i] = self.xmean + self.sigma * arx[i]
            fitness[i] = np.clip(fitness[i], self.lower, self.upper)
            value = self.f.evaluate(func_num, fitness[i])

            self.eval_times += 1
            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break
                
            fitvals[i] = value
            if fitvals[i] < self.optimal_value:
                self.optimal_value = fitvals[i]
                self.optimal_solution = fitness[i]
            #print("optimal: {}\n".format( self.get_optimal()[1]))

        if value == "ReachFunctionLimit":
            return
        # sort and update mean
        argx = np.argsort(fitvals)

        self.xmean = self.xmean + self.sigma * np.sum(self.weights[i] * arx[argx[i]] for i in range(self.mu))

        # update evolution path
        zz = np.sum(self.weights[i] * arz[argx[i]] for i in range(self.mu))
        c = np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps -= self.cs * self.ps
        self.ps += c * zz
        dd = np.sum(self.weights[i] * arx[argx[i]] for i in range(self.mu))
        c = np.sqrt(self.cc * (2 - self.cc) * self.mueff)
        self.pc -= self.cc * self.pc
        self.pc += c * dd

        # update covariance matrix C
        part1 = (1 - self.c1 - self.cmu) * self.C
        part2 = self.c1 * np.dot(self.pc.reshape(self.nn, 1), self.pc.reshape(1, self.nn))
        part3 = np.zeros((self.nn, self.nn))
        for i in range(self.mu):
            part3 += self.cmu * self.weights[i] * np.dot(arx[argx[i]].reshape(self.nn, 1), arx[argx[i]].reshape(1, self.nn))
        self.C = part1 + part2 + part3

        # update step-size
        self.sigma *= np.exp((self.cs / 2) * (np.sum(np.power(x, 2) for x in self.ps) / self.nn - 1))

    def run(self, FES):
        while self.eval_times < FES:
            self.step()

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op =  CMAES_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.nn):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 