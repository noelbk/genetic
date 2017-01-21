#! /usr/bin/env python
# Copyright 2016 Noel Burton-Krahn <noel@burton-krahn.com>

import unittest
import genetic
import random

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestGenetic(unittest.TestCase):
    def setUp(self):
        random.seed(4)
        self.solver = genetic.GeneticSolver()

    def test_child_paths(self):
        random.seed(4)
        f = self.solver.randfunc(4, 4)
        self.assertEqual('(div (div (min 2.12130082004 (max var0 var0)) 2.45321877362) var1)', str(f))
        self.assertEquals(
            ((),
             (0,),
             (0, 0),
             (0, 0, 0),
             (0, 0, 1),
             (0, 0, 1, 0),
             (0, 0, 1, 1),
             (0, 1),
             (1,)),
            tuple(f.child_paths()))

    def test_randfunc(self):
        for arity in range(10):
            f = self.solver.randfunc(arity)
            self.assertEqual(arity, f.arity)
            inputs = self.solver.randinputs(arity, 10)
            for args in inputs:
                f(*args)

    def test_mutate(self):
        f1 = genetic.GeneticConst(self.solver, 3, 1)
        f2 = f1.mutate()
        self.assertNotEqual(str(f1), str(f2))

        for arity in range(10):
            f1 = self.solver.randfunc(arity)
            g1 = self.solver.randfunc(arity)
            f2 = f1.mutate()
            self.assertNotEqual(str(f1), str(f2), "foo")

    def test_survival(self):
        def f(a,b,c):
            return min(2*a,b,c)
        best = genetic.approx_func(3, f)
        log.debug("best match: %s", best)
