#! /usr/bin/env python
# Copyright 2016 Noel Burton-Krahn <noel@burton-krahn.com>

import unittest
import snake

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

     #        "1122#     ",
     #        "1122# C   ",
     #        "1122# c   ",
     #        "11A3####  ",
     #        "11a3333bbb",
     #        "11a3333b b",
     #        "11a3333b b",
     #        "11aaa33b b",
     #        "11##333B b",
     #        "111333bbbb",

     #    2   
     #   222  
     #  1 2 4 
     # 111A444
     #  11a44 
     #   1 4  




class TestSnake(unittest.TestCase):
    def test_loads(self):
        board = snake.Board(10, 10)
        boardstr = (
            "0123456789\n"
            "    # C   \n"
            "    # c   \n"
            "  A ####  \n"
            "  a    bbb\n"
            "  a  * b b\n"
            "  a    b b\n"
            "  aaa  b b\n"
            "  ##   B b\n"
            "      bbbb\n"
            )
        
        board.loads(boardstr)
        self.assertEqual(boardstr, board.dumps())
        
        
        found = list(board.find('A'))
        self.assertEqual(found, [(2,3,32)])
        x, y, head = found[0]

        death = board.smell(board.isdeath)
        food = board.smell(board.isfood)
        gold = board.smell(board.isgold)
        for move in board.adjacent[head]:
            args = [smell[move] for smell in smells]
            score = board.score(*args)
