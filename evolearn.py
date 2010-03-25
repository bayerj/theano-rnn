#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-


from __future__ import division


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import glob
import itertools
import optparse
import os
import sys

import scipy
from pybrain.optimization import PGPE

import rnn



def make_optparse():
  parser = optparse.OptionParser()
  parser.add_option('--datapath', dest='datapath', type='str',
                    help='specify the directory where the data lies.')
  parser.add_option('--hidden', dest='n_hidden', type='int',
                    help='specify number of hiddens to use')
  parser.add_option('--maxevals', dest='maxevals', type='int', default=100,
                    help='specify number of maximum passes through data')
  return parser


def load_dataset(path):
  sortedfilesbyglob = lambda x: sorted(glob.glob(os.path.join(path, '%s*' % x)))
  inptfiles = sortedfilesbyglob('input')
  targetfiles = sortedfilesbyglob('target')

  data = []
  for infn, targetfn in itertools.izip(inptfiles, targetfiles):
    inpt = scipy.loadtxt(infn)
    target = scipy.loadtxt(targetfn)
    target.shape = scipy.size(target), 1
    data.append((inpt, target))
  return data


def make_objective_func(net, data, errorfunc):
  def obj(x):
    net.parameters[:] = x
    error = 0
    for inpt, target in data:
      _, output = net(inpt)
      error += float(errorfunc(output, target))
    return error
  return obj


def stats(net, data):
  true_positives = 0
  total = 0
  for inpt, target in data:
    _, output = net(inpt)
    output = (output > 0.5).astype('float64')
    total += target.shape[0]
    true_positives += (output * target).sum()
  return true_positives, total


def main():
  options, args = make_optparse().parse_args()
  print "Loading data"
  data = load_dataset(options.datapath)

  print "Building network"
  net = rnn.RecurrentNetwork(74, options.n_hidden, 1, outfunc='sig')
  print "Number of parameters:", len(net.parameters)

  objfunc = make_objective_func(net, data, rnn.SumOfSquares())
  x0 = scipy.random.standard_normal(len(net.parameters)) * 0.1
  optimizer = PGPE(objfunc, x0, minimize=True)
  optimizer.maxEvaluations = options.maxevals

  print "First fitness:", objfunc(x0)
  print "Optimizing..."
  params, fitness = optimizer.learn()
  print "Last fitness:", fitness

  true_positives, total = stats(net, data)
  print "Total positives found: %i (%.2f)" % (true_positives, true_positives /
                                                total)
  
  return 0


if __name__ == '__main__':
  sys.exit(main())

