import operator as op
import random
import logging
import math
from genetic.util import DEFAULT, isdefault

LOG = logging.getLogger(__name__)

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
        return self.solver.randfunc(self.arity).simplify()

    def mutate(self, newfunc=None):
        """return a new version of myself with a random mutation"""
        if not newfunc:
            #newfunc = randfunc(self.arity, random.randrange(max(2 * self.height, 1)))
            newfunc = self.solver.randfunc(self.arity, 2)
        return self.subtree().mutate(newfunc).simplify()

    def combine(self, other):
        """combine myself with another function.  Take a random
        portion of myself and swap it with a random portion of other"""
        f = self.subtree()
        g = other.subtree()
        return f.mutate(g.func()), g.mutate(f.func())

    def simplify(self):
        return self

    def is_constant(self):
        return False


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

    def is_constant(self):
        return all(child.is_constant() for child in self.children)
    
    def simplify(self):
        is_constant = True
        simplified = []

        for child in self.children:
            child = child.simplify()
            if not child.is_constant():
                is_constant = False
            simplified.append(child)
        self.children = simplified
        if is_constant:
            try: 
                const = self.op(*[child() for child in self.children])
                return GeneticConst(self.solver, self.arity, const)
            except ValueError:
                return self
        else:
            return self

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

    def is_constant(self):
        return True

class FuncWithErr(object):
    """helper to group functions and fitness"""
    def __init__(self, func):
        self.func = func
        self.err = 0

    def __str__(self):
        return "%.4f %s" % (self.err, str(self.func))

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

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
