#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-

"""Module that offers functionality for a simple recurrent neural network via
the theano library."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import scipy
import theano
import theano.tensor as T


class RecurrentNetwork(object):

  # Expressions.

  _i  = T.matrix('input')
  _t  = T.matrix('target')
  _h0 = T.vector('hidden initial')
  _o0 = T.vector('output initial')

  _W_in = T.matrix('in-weights')
  _W_out = T.matrix('out-weights')
  _W_rec = T.matrix('rec-weights')
  _h_bias = T.vector('bias')

  def one_step_maker(self, hiddenfunc='tanh', outfunc='id'):
    """Return a one step expression function with the given transfer
    functions."""
    hiddenfunc = self.transferfuncmap[hiddenfunc]
    outfunc = self.transferfuncmap[outfunc]

    def one_step(i_t, h_tm1, o_tm1, h_bias, W_in, W_out, W_rec):
      """Perform one step of a simple recurrent network returning the current
      hidden activations and the output.

      `i_t` is the input at the current timestep, `h_tm1` and `o_tm1` are the
      hidden values and outputs of the previous timestep. `h_bias` is the bias for
      the hidden units. `W_in`, `W_out` and `W_rec` are the weight matrices.

      Transfer functions can be specified via `hiddenfunc` and `outfunc` for the
      hidden and the output layer."""
      hidden_in = theano.dot(W_in, i_t)
      hidden_in += theano.dot(W_rec, h_tm1)
      hidden_in += h_bias
      h_t = hiddenfunc(hidden_in)
      o_t = outfunc(theano.dot(W_out, h_t))
      return [h_t, o_t]

    return one_step

  transferfuncmap = {
    'tanh': T.tanh,
    'id': lambda x: x,
    'sig': T.nnet.sigmoid,
  }

  def __init__(self, num_inpt, num_hidden, num_output,
               hiddenfunc='tanh', outfunc='id'):
    self.num_inpt = num_inpt
    self.num_hidden = num_hidden
    self.num_output = num_output

    self.one_step = self.one_step_maker(hiddenfunc, outfunc)

    self._net_expr = theano.scan(
      fn=self.one_step,
      sequences=[self._i],
      outputs_info=[self._h0, self._o0],
      non_sequences=[self._h_bias, self._W_in, self._W_out, self._W_rec],)

    self.netargs = [self._i, self._h0, self._o0, self._h_bias, self._W_in,
                    self._W_out, self._W_rec]

    (self._h_vals, self._o_vals), self._updates = self._net_expr

    self._net_func = theano.function(self.netargs, [self._h_vals, self._o_vals])

    # Organize all the weights in a single array.
    num_inweights = num_inpt * num_hidden
    num_outweights = num_hidden * num_output
    num_recweights = num_hidden**2
    num_biasweights = num_hidden
    self.num_weights = (num_inweights + num_outweights + num_recweights +
                        num_biasweights)

    # We need to fight an index battle here.
    self.parameters = scipy.random.standard_normal(self.num_weights)
    in_start = 0
    out_start = in_start + num_inweights
    rec_start = out_start + num_outweights
    bias_start = rec_start + num_recweights

    self.inweights = self.parameters[in_start:out_start].reshape(
      (num_hidden, num_inpt))
    self.outweights = self.parameters[out_start:rec_start].reshape(
      (num_output, num_hidden))
    self.recweights = self.parameters[rec_start:bias_start].reshape(
      (num_hidden, num_hidden))
    self.biasweights = self.parameters[bias_start:]

  def __call__(self, inpts, initialhidden=None, initialout=None):
    if initialhidden is None:
      initialhidden = scipy.zeros((self.num_hidden, ))
    if initialout is None:
      initialout = scipy.zeros((self.num_output, ))
    return self._net_func(inpts, initialhidden, initialout, self.biasweights,
                          self.inweights, self.outweights, self.recweights)


class ErrorFunction(object):

  output = T.matrix('output')
  target = T.matrix('target')

  def __init__(self):
    self.func = theano.function([self.output, self.target], self.expr)

  def __call__(self, output, target):
    return self.func(output, target)


class SumOfSquares(ErrorFunction):

  expr = 0.5 * T.sum(T.sqr(ErrorFunction.output - ErrorFunction.target))


class FixedTrueNegative(ErrorFunction):

  L = 0.4
  output, target = ErrorFunction.output, ErrorFunction.target
  expr = (1 - target) * T.log(1 - output) - L * target * (1 - output) 
