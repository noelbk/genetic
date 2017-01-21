#! /usr/bin/env python
# Copyright 2016 Noel Burton-Krahn <noel@burton-krahn.com>

import operator as op
import random
import logging
import math
log = logging.getLogger(__name__)

class DEFAULT(): pass
def isdefault(v):
    return v is DEFAULT

class GeneticSolver(object):
    def __init__(self):
        self.OPS = (
            (op.mul, 2),
            (op.div, 2),
            (op.add, 2),
            (op.sub, 2),
            (min, 2),
            (max, 2),
            (op.neg, 1),
            (math.sin, 1),
            (math.exp, 1),
            (math.cos, 1),
            (math.tan, 1),
            )

        self.PROB_OP = 0.75
        self.PROB_VAR = 0.5
        self.PROB_MUTATE = 0.2
        self.RAND_CONST_MIN = -4
        self.RAND_CONST_MAX = 4
        self.RAND_MAXDEPTH = 5

        self.PROB_SUBTREE = 0.50
        self.SUBTREE_CHOOSE_CHILD = False

        self.POPULATION = 40
        self.MAXGENS = 10000
        self.STALE_GENS = 50
        self.REFRESH_GENS = 2*self.STALE_GENS
        self.EPSILON = 1e-3
        self.SURVIVOR_IDXS = [0, 1, 3, 5, 8]
        self.RANDOM_SURVIVORS = 3
        self.SURVIVORS = 5
        self.KEEP_SURVIORS = True

        self.gen_count = 0

    def randfunc(self, arity, maxdepth=DEFAULT):
        """return a random function with arity (number of input
        arguments).  Maxdepth is the max depth of the function evaluation
        tree"""
        if isdefault(maxdepth):
            maxdepth = self.RAND_MAXDEPTH
        if maxdepth > 0 and random.random() < self.PROB_OP:
            op, op_arity = random.choice(self.OPS)
            return GeneticOp(self, arity, op, [self.randfunc(arity, maxdepth-1) for i in range(op_arity)])

        if arity > 0 and random.random() < self.PROB_VAR:
            return GeneticVar(self, arity, random.randrange(arity))
        else:
            return GeneticConst(self, arity, random.uniform(self.RAND_CONST_MIN, self.RAND_CONST_MAX))

    def population(self, arity, size=DEFAULT):
        """return an initial population of random functions"""
        if isdefault(size):
            size = self.POPULATION
        return [FuncWithErr(self.randfunc(arity, self.RAND_MAXDEPTH)) for i in range(size)]

    def generation(self, funcs, survivors=DEFAULT):
        """make a new generation of functions.  Breed the rest from the top survivors"""
        if isdefault(survivors):
            survivors = self.SURVIVORS
        size = len(funcs)

        parents = sorted(funcs, key=lambda f: f.err)
        self.gen_count += 1
        for i in range(1):
            print "generation[%4d]: %s%s\r" % (self.gen_count, " " * i, str(parents[i])[:60]),
            #log.debug("generation[%4d]: %s%s", self.gen_count, " " * i, str(parents[i])[:60])

        if hasattr(self, 'SURVIVOR_IDXS'):
            parents = [parents[i] for i in self.SURVIVOR_IDXS]
            survivors = len(self.SURVIVOR_IDXS)
        else:
            if survivors < 1:
                survivors = int(len(funcs) * survivors)
            parents = parents[:survivors]

        if getattr(self, 'KEEP_SURVIORS', False):
            nextgen = parents[:survivors]
        else:
            nextgen = []

        if hasattr(self, 'RANDOM_SURVIVORS'):
            n = self.RANDOM_SURVIVORS
            if n < 1:
                n = int(survivors * self.RANDOM_SURVIVORS)
            for i in range(n):
                parents.append(FuncWithErr(parents[0].func.randnew()))

        while len(nextgen) < size:
            fidx = random.randrange(len(parents))
            f = parents[fidx].func

            if random.random() < self.PROB_MUTATE:
                h = f.mutate()
                nextgen.append(FuncWithErr(h))
            else:
                gidx = random.randrange(len(parents))
                if gidx == fidx:
                    gidx = (gidx + 1) % len(parents)
                g = parents[gidx].func
                for h in f.combine(g):
                    nextgen.append(FuncWithErr(h))
        return parents, nextgen

    def evolve(self, pop, fitness, maxgens=DEFAULT, eps=DEFAULT):
        """evolve pop over maxgens generations"""
        for genidx in range(maxgens):
            for indv in pop:
                indv.err += fitness(indv.func)
            parents, pop = self.generation(pop)
            if parents[0].err < eps:
                break
        return parents, pop

    def survival(self, initpop, fitness, maxgens=DEFAULT, eps=DEFAULT):
        """try to generate the best survivor that matches the inputs"""
        if isdefault(maxgens):
            maxgens = self.MAXGENS
        if isdefault(eps):
            eps=self.EPSILON

        pop = [FuncWithErr(i) for i in initpop]
        size = len(initpop)

        if not getattr(self, 'STALE_GENS', 0):
            return self.evolve(pop, fitness, maxgens, eps)

        self.gen_count = 0
        while self.gen_count < maxgens:
            parents, pop = self.evolve(pop, fitness, self.STALE_GENS, eps)
            if self.gen_count >= maxgens or parents[0].err < eps:
                break

            log.debug("got stale, refreshing...")

            # try evolving a random pop to see if that can shake up stale pop
            others = [FuncWithErr(initpop[0].randnew()) for i in range(size)]
            parents, others = self.evolve(others, fitness, self.REFRESH_GENS, eps)
            if parents[0].err < eps:
                pop = others
                break

            # mix current pop with others
            pop = [random.choice((pop[i], others[i])) for i in range(size)]

        return parents, pop


    def randinputs(self, argranges, ninputs):
        """generate a random list of inputs"""
        try:
            iter(argranges)
        except TypeError:
            argranges = tuple((-4, 4) for i in range(argranges))
        return tuple(tuple(random.uniform(a,b) for a,b in argranges) for i in range(ninputs))

    def tournament(self, game):
        """Keep running game on subsets of the population, keeping the winners"""
        assert False, "unimplemented"


class GeneticFunc(object):
    """Base class for genetic functions.  Functions can be evaluated
    with 1..arity arguments, mutated, and combined"""
    def __init__(self, solver, arity):
        self.solver = solver
        self.arity = arity
        self.children = None

    @property
    def height(self):
        return 0

    def __call__(self, *args):
        assert false, "unimplemented"

    def __str__(self):
        assert false, "unimplemented"

    def clone(self):
        return self

    def child_count(self):
        return 1

    def child_paths(self, path=()):
        yield path

    def subtree(self):
        return GeneticFuncSubtree(self)

    def randnew(self):
        return self.solver.randfunc(self.arity)

    def mutate(self, newfunc=None):
        """return a new version of myself with a random mutation"""
        if not newfunc:
            #newfunc = randfunc(self.arity, random.randrange(max(2 * self.height, 1)))
            newfunc = self.solver.randfunc(self.arity, 2)
        return self.subtree().mutate(newfunc)

    def combine(self, other):
        """combine myself with another function.  Take a random
        portion of myself and swap it with a random portion of other"""
        f = self.subtree()
        g = other.subtree()
        return f.mutate(g.func()), g.mutate(f.func())

class GeneticOp(GeneticFunc):
    """A function with arity child sub-functions"""
    def __init__(self, solver, arity, op, children):
        super(GeneticOp, self).__init__(solver, arity)
        self.op = op
        self.children = children

    def __str__(self):
        return '(' + ' '.join((self.op.__name__,) + tuple(str(child) for child in self.children)) + ')'

    def __call__(self, *args):
        return self.op(*[child(*args) for child in self.children])

    def child_count(self):
        return 1 + sum([child.child_count() for child in self.children])

    @property
    def height(self):
        return 1 + max([child.height for child in self.children])

    def child_paths(self, path=()):
        yield path
        for idx in range(len(self.children)):
            for p in self.children[idx].child_paths(path + (idx,)):
                yield p

    def subtree(self):
        if getattr(self, 'SUBTREE_CHOOSE_CHILD', False):
            child_path = random.choice(tuple(self.child_paths()))
        else:
            child_path = []
            child = self
            while child and child.children:
                child_idx = random.randrange(len(child.children))
                child_path.append(child_idx)
                child = child.children[child_idx]
                #if child.height == 0 or random.random() < 1/(child.height+1):
                if child.height == 0 or random.random() < self.solver.PROB_SUBTREE:
                    break
        return GeneticFuncSubtree(self, child_path)

    def clone(self):
        return GeneticOp(self.solver, self.arity, self.op, [child.clone() for child in self.children])

class GeneticFuncSubtree(object):
    """A helper class to mutate and replace parts of a GenerticOp tree"""
    def __init__(self, root, child_path=None):
        self.root = root
        self.child_path = child_path or []

    def func(self):
        child = self.root
        for child_idx in self.child_path:
            child = child.children[child_idx]
        return child.clone()

    def mutate(self, func):
        if not self.child_path:
            return func.clone()

        newfunc = self.root.clone()
        child = newfunc
        for child_idx in self.child_path[:-1]:
            child = child.children[child_idx]
        child.children[self.child_path[-1]] = func
        return newfunc

class GeneticVar(GeneticFunc):
    """returns the value of one argument"""
    def __init__(self, solver, arity, idx):
        super(GeneticVar, self).__init__(solver, arity)
        self.idx = idx

    def __str__(self):
        return 'var%d' % self.idx

    def __call__(self, *args):
        return args[self.idx]

class GeneticConst(GeneticFunc):
    """return a constant"""
    def __init__(self, solver, arity, const):
        super(GeneticConst, self).__init__(solver, arity)
        self.const = const

    def __str__(self):
        return str(self.const)

    def __call__(self, *args):
        return self.const

class FuncWithErr(object):
    """helper to group functions and fitness"""
    def __init__(self, func):
        self.func = func
        self.err = 0

    def __str__(self):
        return "%.4f %s" % (self.err, str(self.func))

def approx_func(arity, func, maxgens=DEFAULT, eps=DEFAULT):
    solver = GeneticSolver()

    inputs = solver.randinputs(arity, 10)
    expect = list([func(*args) for args in inputs])

    def fitness(test_func):
        err = 0

        # sum of squares of (expect - test_func(inputs))
        for args, expected in zip(inputs, expect):
            try:
                delta = expected - test_func(*args)
            except:
                delta = float("inf")
            err += delta * delta

        # the fatter the un-fitter!  multiply un-fitness by the number of child nodes
        err *= max(test_func.child_count() - 2 ** arity, 1)
        return err

    pop = [ solver.randfunc(arity) for i in range(solver.POPULATION) ]
    parents, funcs = solver.survival(pop, fitness, maxgens, eps)
    log.debug("\nbest match: %s", parents[0])
    log.debug("\n\n")
    return parents[0]

def test():
    approx_func(0, lambda: math.pi * 10)
    approx_func(1, lambda a: math.sin(a))
    approx_func(1, lambda a: -a*a)
    approx_func(2, lambda a, b: max(a-b, a*b))
    approx_func(3, lambda a, b, c: a*a - min(b,c))
    approx_func(4, lambda a, b, c, d: a*b + d*d - c*a*b)
    approx_func(2, lambda a, b: math.sin(a) + math.cos(min(a,b)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test()
