#! /usr/bin/env python
# Copyright 2016 Noel Burton-Krahn <noel@burton-krahn.com>

import unittest
import genetic
import random

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestParser(unittest.TestCase):
    def setUp(self):
        random.seed(4)
        self.solver = genetic.GeneticSolver()

    def test_parser_simple(self):
        funcstr = "(add var0 var1)"
        parsed = self.solver.parsefunc(2, funcstr)
        self.assertEqual(funcstr, str(parsed))
        self.assertEqual(3, parsed(1, 2))

    def test_parser(self):
        random.seed(4)
        f = self.solver.randfunc(4, 4)
        parsed = self.solver.parsefunc(4, str(f))
        self.assertEqual(str(f), str(parsed))
        args = (1,2,3,4)
        eps = abs(f(*args) - parsed(*args))
        self.assertTrue(eps < 1e-6, "Error too big! err=%s parsed=%s" % (eps, parsed))
