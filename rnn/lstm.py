"""Module that contains functionality for lstm RNNs."""

import theano
import theano.tensor as T


lstminpt = T.vector('lstm-input')
state = T.vector('lstm-state')

slicesize = lstminpt.shape[0] / 4
inpt = lstminpt[:slicesize]
ingate = lstminpt[slicesize: 2 * slicesize]
forgetgate = lstminpt[2 * slicesize:3 * slicesize]
outgate = lstminpt[3 * slicesize:4 * slicesize]

ingate = T.nnet.sigmoid(ingate)
forgetgate = T.nnet.sigmoid(forgetgate)
outgate = T.nnet.sigmoid(outgate)

new_state = inpt * ingate + state * forgetgate
output = new_state * outgate

lstm = theano.function([lstminpt, state], [new_state, output])
