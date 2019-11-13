#Program assigns probability to nodes. 
import random
import json
import time
import numpy

#copys input dict and adds x: y to the new dict and returns the new dict
def extend(input_dict, x, y):
    l = input_dict.copy()
    l[x] = y
    return l

#enumeration is exact, mcmc and likelihood_weighting are estimates
#X is the first variable ex: nodes[0].id
#events is a dictionary of preassaigned events
#bn is list of nodes
def enumeration_ask(X, events, bn, vars):
    assert X not in events
    Q = {}
    for xi in range(0, 2):# goes from 0 to 1 inclusive. true-false
        Q[xi] = enumerate_all(vars, extend(events, X, xi), bn) 
        #passes in node numbers in topo order, evidence/events + current X state in loop, and list of nodes
    return normalize(Q) #normallizes the results

#recursive helper function for enumeration ask
def enumerate_all(vars, events, bn):
    if not vars: return 1.0
    Y, rest = vars[0], vars[1:] #seperates first node id in topo sorted list
    Ynode = bn[Y] #grabs Y from list of nodes
    if Y in events:
        return Ynode.p(events[Y], events) * enumerate_all(rest, events, bn) 
        #multiplies prob(Y|parents(Y)) * enumerate_all(all the other nodes, the same evidence passed in, the list of nodes )
    else:
        return sum(Ynode.p(y, events) * enumerate_all(rest, extend(events, Y, y), bn) for y in range(0,2))# y = 0 then y = 1
        #returns the sum of ( P(Y = 0| parents(Y) * enum_all(all other nodes, evidence + Y = 0, list of nodes) 
        # + P(Y = 1| parents(Y) * enum_all(all other nodes, evidence + Y = 1, list of nodes) ))

#estimates the probability with a likelihood function
#X: query variable
#e: evidence
#bn: list of nodes
#order: topological ordering of nodes
#n: number of computations. more computations is more accurate
def likelihood_weighting(X, e, bn, order, N = 1000):
    W = {0: 0, 1: 0}
    for _ in range(N):
        sample, weight = weighted_sample(bn,e, order)
        W[sample[X]] += weight
    return normalize(W)
#helper function for likelihood_weighting
def weighted_sample(bn, e, order):
    w = 1
    x = dict(e)
    for idx in order:
        if bn[idx].id in e:
            w = w * bn[idx].p(e[bn[idx].id], x)
        else:
            x[bn[idx].id] = bn[idx].sample(bn, e)
    return x,w

#computes probability of X given events
#X: query variable
#events: evidence
#bn: list of nodes
#order: topological ordering of nodes
#n: number of computations. more computations is more accurate
def mcmc(X, events, bn, order, n = 1000):
    assert X not in events
    N = {0:0, 1:0}
    x = dict(events)
    Z = [v for v in order if v not in events]
    for v in Z:
        x[v] = random.randint(0,1)
    for _ in range(0,n):
        for z in Z:
            x[z] = markov_blanket(z, x, bn)#need to sample from P(z|mb(z))
            N[x[X]] += 1
    return normalize(N)

#mcmc helper function. returns z state and  x state
def markov_blanket(X, e, bn):
    prob = {0:0, 1:0}
    x_node = bn[X]
    for x_i in range(0,2):
        ev = extend(e, X, x_i)
        prob[x_i] = x_node.p(x_i, e) * product( [bn[y].p(ev[y], ev) for y in x_node.children] )
    prob = normalize(prob)
    if random.uniform(0,1) < prob[1]:
        return 1
    else:
        return 0


#node class to store id, parents, and probability
class Node:
    def __init__(self, in_id, input):
        self.id = in_id
        self.parents = input[str(in_id)]['parents']
        self.prob = input[str(in_id)]['prob']
        self.children = []
    
    #samples this node given a dictionary of nodes and evidence
    def sample(self, bn, events):
        if self.id in events:
            return events[self.id]
        if len(self.parents) == 0:
            if random.uniform(0,1) < self.prob[0][1]:
                return 1
            else:
                return 0
        l = []
        for p in self.parents:
            if p not in events:
                l.append(bn[p].sample(bn, events))
            else:
                l.append(events[p])
        if random.uniform(0,1) < self.list_probability(l):
            return 1
        else:
            return 0
    #computes probability given a set of the states of all the parents. helper function
    def list_probability(self, events):
        assert len(events) == len(self.parents)
        for p in self.prob:
            if p[0] == events:
                return p[1]
    #computes probability of self given evidence which should include parents
    def p(self, state, evidence):
        e = [v for x, v in evidence.items() if x in self.parents]
        pr = self.list_probability(e)
        assert isinstance(pr, float)
        return pr if state == 1 else 1 - pr
    
    def set_children(self, bn):
        for x in bn:
            if self.id in x.parents:
                self.children.append(x.id)

    def __repr__(self):
        return str(self.id)

#normalize computes the normal of a dictionary
def normalize(d):
    some = 0
    for _, val in d.items():
        some += val
    if some == 0:
        return d
    return {key: value/some for key, value in d.items()}

#multiplies everything in the list. if list is empty, return 1
def product(l):
    if not l:
        return 1
    total = 1
    for item in l:
        total*=item
    return total
    
#Topological sort I used for another class. 
def topsort(G):
    count = dict((u, 0) for u in G)#The in-degree for each node
    for u in G:
        for v in G[u]:
            count[v] += 1    #Count every in-edge
    Q = [u for u in G if count[u] == 0] # Valid initial nodes
    S = []                   #The result
    while Q:                 #While we have start nodes...
        u = Q.pop()          #Pick one
        S.append(u)          #Use it as first of the rest
        for v in G[u]:
            count[v] -= 1    #"Uncount" its out-edges
            if count[v] == 0:#New valid start nodes?
                Q.append(v)  #Deal with them next
    return S


if __name__ == '__main__':
    bn = []
    with open('./bn.json') as f:
        bn = json.load(f)
    #print(bn)
    nodes=[]
    d_nodes = {}
    for node in bn:
        n = Node(len(nodes), bn)
        nodes.append(n)
        d_nodes[n.id] = n.parents

    for n in nodes: #sets children for use in MCMC alg
        n.set_children(nodes)

    order = topsort(d_nodes)#gets a topological ordering of nodes
    order.reverse() #since it is in respect to children not parents this puts it in correct order.
    
    
    enum_time = []
    likely_time = []
    mcmc_time = []
    enum_result = []
    likely_result = []
    mcmc_result = []

    for i in range(len(nodes)):

        for k in range(0,len(nodes)):
            for j in range(0,2):
                if k == i:
                    continue
                e = {i:j}#evidence set
                start = time.time()
                enum_result.append( enumeration_ask(nodes[k].id, e, nodes, order) )
                end = time.time()
                enum_time.append(end - start)

                start = time.time()
                likely_result.append(likelihood_weighting(nodes[k].id, e, nodes, order) )
                end = time.time()
                likely_time.append( end - start)

                start = time.time()
                mcmc_result.append( mcmc(nodes[k].id, e, nodes, order) )
                end = time.time()
                mcmc_time.append( end - start )
    
    #calculates mean squared error
    likely_mse = 0
    mcmc_mse = 0
    for i in range(len(likely_result)):
        likely_mse += (likely_result[i][1] - enum_result[i][1]) **2
        mcmc_mse += (mcmc_result[i][1] - enum_result[i][1] )**2
    mcmc_mse = mcmc_mse/len(mcmc_time)
    likely_mse = likely_mse/len(likely_time)

    #calculates average run time
    mcmc_avg = 0
    likely_avg = 0
    enum_avg = 0
    for i in range(len(mcmc_time)):
        mcmc_avg +=mcmc_time[i]
        likely_avg += likely_time[i]
        enum_avg += enum_time[i]
    mcmc_avg /= len(mcmc_time)
    likely_avg /= len(likely_time)
    enum_avg /= len(enum_time)
    
    print("avg times: enum: ", enum_avg)
    print( 'likely: ', likely_avg)
    print('mcmc: ', mcmc_avg)
    print("mcmc RMSE: ", mcmc_mse**(.5))
    print('likely RMSE: ', likely_mse**(.5))
    