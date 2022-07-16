'''

@file parser.py
Defines Tableau calculus as a lexer and parser.

'''
import re

from tdparser import Lexer, Token


class Symbol(Token):

    def __init__(self, text):
        # self.value = int(text)
        pass

    def nud(self, context):
        '''
        What the token's value is
        '''
        pass


class And(Token):
    lbp = 10  # Precendence
    pass

    def led(self, left, context):
        '''
        Compute the value of this token 
        when between two expressions'''


class Or(Token):
    pass


class Not(Token):
    pass
