import operator as op
import random
import math
import logging
import re

from pyparsing import OneOrMore, nestedExpr
from genetic.util import DEFAULT, isdefault
from genetic.funcs import GeneticOp, GeneticConst, GeneticVar, FuncWithErr

LOG = logging.getLogger(__name__)

def if_neg(a, b, c):
    return b if a <= 0 else c

def sin2pi(t):
    return math.sin(2*math.pi * t)

def cos2pi(t):
    return math.cos(2*math.pi * t)

def square(t):
    return t * t

def sigmoid(t):
    return t / math.sqrt(1+t*t)

class GeneticSolver(object):
    def __init__(self):
        self.OPS = (
            (if_neg, 3),
            (op.mul, 2),
            (op.div, 2),
            (op.add, 2),
            (min, 2),
            (op.neg, 1),
            (sin2pi, 1),
            (math.exp, 1),
            )
        self.OPS_byname = {op.__name__: (op, op_arity) for op, op_arity in self.OPS}
        # for backwards compatibility
        self.OPS_byname["sin"] = (math.sin, 1)
        self.OPS_byname["cos"] = (math.cos, 1)
        self.OPS_byname["max"] = (max, 2)
        self.OPS_byname["cos2pi"] = (cos2pi, 1)
        self.OPS_byname["square"] = (square, 1)
        self.OPS_byname["log"] = (math.log, 1)
        self.OPS_byname["sigmoid"] = (sigmoid, 1)

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
        #self.SURVIVOR_IDXS = [0, 1, 3, 5, 8]
        self.SURVIVORS = 5
        self.RANDOM_SURVIVORS = 1
        self.KEEP_SURVIORS = 1

        self.gen_count = 0

    def parsefunc(self, arity, funcstr):
        if funcstr[0] != '(':
            funcstr = "(" + funcstr + ")"
        parsed = OneOrMore(nestedExpr()).parseString(funcstr)
        return self._parsefunc_list(parsed[0].asList(), arity)

    re_var = re.compile(r'var(\d+)')
    def _parsefunc_list(self, expr, arity):
        if type(expr) is list:
            op_name = expr[0]
            if op_name in self.OPS_byname:
                op, op_arity = self.OPS_byname[op_name]
                assert len(expr)-1 == op_arity, "bad arity for op=%s.  Expected %s args, got %s" % (op_name, op_arity, expr)
                args = [self._parsefunc_list(arg, arity) for arg in expr[1:]]
                return GeneticOp(self, arity, op, args)
            elif len(expr) == 1:
                return self._parsefunc_list(expr[0], arity)
            else:
                assert False, "unrecognized operator: %s.  expected one of %s" % (expr[0], (op for op, op_arity in self.OPS_byname))
            
        elif expr[0] == '"':
            assert expr[-1] == '"', 'string constants must begin and end with "" got=[%s]' % expr
            return GeneticConst(self, arity, expr[1:-1].encode('string_escape'))
        
        elif expr[0].isdigit() or expr[0] in "+-.":
            return GeneticConst(self, arity, float(expr))
        
        else:
            m = self.re_var.match(expr)
            assert m, "unrecognized variable name=%s, expected something like var0..var%s" % (expr, arity-1)
            idx = int(m.group(1))
            assert 0 <= idx < arity, "variable index=%s out of bounds for arity=%s" % (idx, arity)
            return GeneticVar(self, arity, idx)
            
    
        
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

    def start(self, maxgens=DEFAULT, eps=DEFAULT):
        """try to generate the best survivor that matches the inputs"""
        if isdefault(maxgens):
            maxgens = self.MAXGENS
        if isdefault(eps):
            eps=self.EPSILON
        self.maxgens = maxgens
        self.eps = eps
        self.gen_count = 0
        
    def converged(self, err=None):
        # TODO - return True when error flattens out, not just after maxits
        return (self.maxgens and self.gen_count > self.maxgens) or (err is not None and err < self.eps)

    def generation(self, funcs, survivors=DEFAULT):
        """make a new generation of functions.  Breed the rest from the top survivors"""
        if isdefault(survivors):
            survivors = self.SURVIVORS
        size = len(funcs)

        parents = sorted(funcs, key=lambda f: f.err)
        self.gen_count += 1
        for i in range(1):
            LOG.debug("generation[%4d]: %s%s", self.gen_count, " " * i, str(parents[i])[:60])

        if hasattr(self, 'SURVIVOR_IDXS'):
            parents = [parents[i] for i in self.SURVIVOR_IDXS if i < len(parents)]
            survivors = len(self.SURVIVOR_IDXS)
        else:
            if survivors < 1:
                survivors = int(len(funcs) * survivors)
            parents = parents[:survivors]

        nextgen = parents[:getattr(self, 'KEEP_SURVIORS', 0)]

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
                    if len(nextgen) < size:
                        nextgen.append(FuncWithErr(h))
                    else:
                        break
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

            LOG.debug("got stale, refreshing...")

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
    LOG.debug("\nbest match: %s", parents[0])
    LOG.debug("\n\n")
    return parents[0]

