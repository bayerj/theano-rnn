#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import optparse
import sys
import time

import scipy

from rnn import RecurrentNetwork, LstmNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LSTMLayer, TanhLayer


def make_optparse():
  parser = optparse.OptionParser()
  parser.add_option('--lstm', dest='lstm', action='store_true')
  return parser


def theano(num_inpt, num_hidden, num_output, inpts, lstm=False):
  klass = LstmNetwork if lstm else RecurrentNetwork
  rnn = klass(num_inpt, num_hidden, num_output)

  start = time.time()
  for inpt in inpts:
    rnn(inpt)
  return time.time() - start


def pybrain(num_inpt, num_hidden, num_output, inpts, lstm=False):
  net = buildNetwork(num_inpt, num_hidden, num_output, recurrent=True,
                     hiddenclass=LSTMLayer if lstm else TanhLayer,
                     fast=False)
  start = time.time()
  for seq in inpts:
    net.reset()
    for inpt in seq:
      net.activate(inpt)
  return time.time() - start


def pybrainarac(num_inpt, num_hidden, num_output, inpts, lstm=False):
  net = buildNetwork(num_inpt, num_hidden, num_output, recurrent=True,
                     hiddenclass=LSTMLayer if lstm else TanhLayer,
                     fast=True)
  start = time.time()
  for seq in inpts:
    net.reset()
    for inpt in seq:
      net.activate(inpt)
  return time.time() - start


def main():
  options, args = make_optparse().parse_args()
  num_inpt = int(args[0])
  num_hidden = int(args[1])
  num_output = int(args[2])

  print "Network stats"
  print "-" * 20
  print "Number of inputs: %i" % num_inpt
  print "Number of hidden: %i" % num_hidden
  print "Number of outputs: %i" % num_output

  inpts = scipy.random.random((500, 100, num_inpt))

  print
  print "Durations"
  print "-" * 20
  pybrain_dur = pybrain(num_inpt, num_hidden, num_output, inpts,
                        lstm=options.lstm)
  print "Pybrain: %.2f" % pybrain_dur

  pybrainarac_dur = pybrainarac(num_inpt, num_hidden, num_output, inpts,
                                lstm=options.lstm)
  print "Pybrain \w arac: %.2f" % pybrainarac_dur

  theano_dur = theano(num_inpt, num_hidden, num_output, inpts, lstm=options.lstm)
  print "Theano: %.2f" % theano_dur

  print
  print "Ratios"
  print "-" * 20

  print "Theano / PyBrain: %.2f" % (theano_dur / pybrain_dur)
  print "Theano / PyBrain+arac: %.2f" % (theano_dur / pybrainarac_dur)
  print "PyBrain+arac / PyBrain: %.2f" % (pybrainarac_dur / pybrain_dur)

  return 0


if __name__ == '__main__':
  sys.exit(main())

