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
        random.seed(1)

    def test_child_paths(self):
        f = genetic.randfunc(4, 4)
        self.assertEqual('(max var1 (min var0 var3))', str(f))
        self.assertEquals(((), (0,), (1,), (1, 0), (1, 1)),
                          tuple(f.child_paths()))
        
        f = genetic.randfunc(4, 4)
        self.assertEqual('(sub (div -3.75528013573 (sub var0 (mul var1 var0)))'
                         ' (div (add (max 1.13835490346 var3) var1)'
                         ' (min var3 (add 3.05983200665 0.0422705646368))))', str(f))
        self.assertEquals(((), (0,), (0, 0), (0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 1, 0),
                           (0, 1, 1, 1), (1,), (1, 0), (1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1),
                           (1, 0, 1), (1, 1), (1, 1, 0), (1, 1, 1), (1, 1, 1, 0), (1, 1, 1, 1)),
                          tuple(f.child_paths()))

        
    def test_randfunc(self):
        for arity in range(10):
            f = genetic.randfunc(arity)
            self.assertEqual(arity, f.arity)
            inputs = genetic.randinputs(arity, 10)
            for args in inputs:
                f(*args)
        
    def test_mutate(self):
        f1 = genetic.GeneticConst(3, 1)
        f2 = f1.mutate()
        self.assertNotEqual(str(f1), str(f2))
        
        for arity in range(10):
            f1 = genetic.randfunc(arity)
            g1 = genetic.randfunc(arity)
            f2 = f1.mutate()
            self.assertNotEqual(str(f1), str(f2), "foo")
        
    def test_approximate(self):
        def f(a,b,c):
            return min(2*a,b,c)
        inputs = genetic.randinputs(3, 100)
        funcs = genetic.approximate(f, inputs, maxgens=10)
        log.debug("best match: %s", funcs[0])
                  
        
    
    
