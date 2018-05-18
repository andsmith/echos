import numpy as np
import pylab as plt


class ESN(object):
    def __init__(self, n, sparsity, spectral_radius, n_in, n_out, out_fn=None, out_fn_inv=None, noise=None, input_scaling=1.0):
        self._n = n
        self._sparse = sparsity
        self._rad = spectral_radius
        self._n_in = n_in
        self._n_out = n_out
        self._out_fn = out_fn
        self._out_fn_inv = out_fn_inv
        self._noise = noise
        self._input_scaling = input_scaling

        self._w_in = np.random.randn(self._n_in * self._n).reshape(self._n, self._n_in)
        self._w_out = None
        self._w = np.random.randn(self._n * self._n)
        self._w[np.random.rand(self._n * self._n) > sparsity] = 0.0
        self._w = self._w.reshape(n,n)
        v, _ = np.linalg.eig(self._w)
        sr = np.max(np.abs(v))
        self._w *= (spectral_radius / sr)

        self.init()

    def get_state(self):
        return self._s

    def init(self, value=None):
        if value is None:
            self._s = np.random.randn(self._n)
        else:
            self._m = np.zeros(self._n) + value

    def tick(self, input=None):
        activations =  np.dot(self._w, self._s)
        if input is not None:
            activations += self._input_scaling * np.dot(self._w_in, input)
        if self._noise is not None:
            activations += self._noise * np.random.randn(self._n)
        self._s = np.tanh(activations)
        return self._s

    def train_classification(self, inputs, outputs, regularization=0.1, burn_in = 100 ):
        N = len(inputs)
        assert N==len(outputs), "Need same number of training inputs as outputs!"
        final_states = []
        for i in range(N):
            self.init()
            for _ in range(burn_in):
                self.tick()
            for row in inputs[i]:
                self.tick(row)
            final_states.append(self.get_state())
        A = np.vstack((np.hstack(final_states), np.ones(N)))
        y = np.vstack(outputs)
        if self._out_fn is not None:
            y = self._out_fn_inv(y)
        self._w_out = np.dot(regularized_pseudoinverse(A, regularization), y)
        y_hat = np.dot(self._w_out, A)
        return y_hat

def regularized_pseudoinverse(x, l):
    internal = np.dot(x.T, x) + l * np.identity(x.shape[1])
    return np.dot(np.linalg.inv(internal), x.T)



if __name__ == "__main__":
    e = ESN(700, .02, .97, 2, 1)
    states = []
    for _ in range(200):
        states.append(e.tick()[:5])
    states = np.vstack(states)
    for row in states.T:
        plt.plot(row)
    plt.show()
