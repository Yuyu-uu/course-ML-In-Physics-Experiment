import tensorflow as tf
from numpy import pi
import numpy as np

from ToComplex import to_complex
from Mask import create_mask
from Propagation import diff
from H import H
# from LensGenerator import lens_generator


def make_variables(layers, k, initializer):
    return tf.Variable(initializer(shape=[layers, k, k], dtype=tf.float32))


class DNN:
    def __init__(self, layers, dim_in, dim_out, phase_dim, lmb, ds, amp_or_phase, phase_init=0.1, expand_times=8):
        assert len(ds) == layers + 1, "Lost diffraction distances"
        assert layers == len(amp_or_phase), "Layers should be equal to len(amp_or_phase)"
        self.layers = layers
        self.dim_in = dim_in            # Layer dimension In
        self.dim_out = dim_out          # Layer dimension Out
        self.phase_dim = phase_dim      # Phase dimension
        self.dim_expanded = int(expand_times * self.dim_in)
        self.amp_or_phase = amp_or_phase

        # The Spectral Propagator with diffraction distance ds
        self.Hs = []
        for d in ds:
            self.Hs.append(H(dim=self.dim_expanded, d=d, lmb=lmb, pixel_size=6.4e-6))

        self.phase = make_variables(layers=layers, k=phase_dim,
                                    initializer=tf.random_normal_initializer(mean=0., stddev=phase_init))


class D2NN(DNN):
    def __init__(self, layers, dim_in, dim_out, phase_dim, lmb, ds, amp_or_phase, phase_init,
                 mask_border=0, rang_size=50, mask_on=False, normalize=False):
        super(D2NN, self).__init__(layers, dim_in, dim_out, phase_dim, lmb, ds, amp_or_phase, phase_init)
        masks, _, _ = create_mask(dim=self.dim_out, border=mask_border, rang_size=rang_size)
        self.mask = tf.math.reduce_sum(masks, axis=0)
        self.mask_on = mask_on
        self.layers = layers
        self.normalize = normalize

    # The Angular Spectrum Approach
    def propagation(self, input_image):
        ts = []
        for i, a_or_p in enumerate(self.amp_or_phase):
            if a_or_p == 'amp':
                ts.append(to_complex(tf.sigmoid(self.phase[i])))
            else:
                ts.append(tf.math.exp(-2j * pi * to_complex(tf.sigmoid(self.phase[i]))))

        m0 = to_complex(input_image)
        # Forward propagation
        for i, t in enumerate(ts):
            m0_diff = diff(m0, self.Hs[i], dim_out=self.phase_dim)
            m0 = tf.math.multiply(m0_diff, t)
        ml = diff(m0, self.Hs[-1], dim_out=self.dim_out)

        # output
        m_out = tf.math.square(tf.math.abs(ml))
        if self.mask_on:
            m_out = m_out * self.mask
        if self.normalize:
            m_out = m_out / tf.reduce_max(m_out * self.mask)

        return m_out

    @tf.function
    def __call__(self, input_image):
        return tf.vectorized_map(fn=self.propagation, elems=input_image, fallback_to_while_loop=True)


class FD2NN(DNN):
    def __init__(self, layers, dim_in, dim_out, phase_dim, lmb, ds, amp_or_phase, phase_init,
                 mask_border=0, rang_size=50, mask_on=False, normalize=False):
        super(FD2NN, self).__init__(layers, dim_in, dim_out, phase_dim, lmb, ds, amp_or_phase, phase_init)
        masks, _, _ = create_mask(dim=self.dim_out, border=mask_border, rang_size=rang_size)
        self.mask = tf.math.reduce_sum(masks, axis=0)
        self.mask_on = mask_on
        self.layers = layers
        self.normalize = normalize

    # The Angular Spectrum Approach
    def propagation(self, input_image, phaseD2NN=None):
        ts = []
        for i, a_or_p in enumerate(self.amp_or_phase):
            if a_or_p == 'amp':
                if phaseD2NN is None:
                    ts.append(to_complex(tf.sigmoid(self.phase[i])))
                else:
                    ts.append(to_complex(tf.sigmoid(phaseD2NN[i])))
            else:
                if phaseD2NN is None:
                    ts.append(tf.math.exp(-2j * pi * to_complex(tf.sigmoid(self.phase[i]))))
                else:
                    ts.append(tf.math.exp(-2j * pi * to_complex(tf.sigmoid(phaseD2NN[i]))))

        m0 = to_complex(input_image)
        m0_f = tf.signal.fftshift(tf.signal.fft2d(m0))
        # Forward propagation
        for i, t in enumerate(ts):
            m0_f_diff = diff(m0_f, self.Hs[i], dim_out=self.phase_dim)
            m0_f = tf.math.multiply(m0_f_diff, t)
        ml_f = diff(m0_f, self.Hs[-1], dim_out=self.dim_out)
        ml = tf.signal.fftshift(tf.signal.fft2d(ml_f))

        # output
        m_out = tf.math.square(tf.math.abs(ml))
        if self.mask_on:
            m_out = m_out * self.mask
        if self.normalize:
            m_out = m_out / tf.reduce_max(m_out * self.mask)
        m_out = tf.reverse(m_out, [0, 1])

        return m_out

    @tf.function
    def __call__(self, input_image, phaseD2NN=None):
        return tf.vectorized_map(fn=self.propagation, elems=(input_image, phaseD2NN), fallback_to_while_loop=True)


class D2NNBinary(D2NN):
    def __init__(self, layers, dim_in, dim_out, phase_dim, lmb, ds,
                 mask_border=0, rang_size=50, mask_on=False, normalize=False):
        super(D2NNBinary, self).__init__(layers, dim_in, dim_out, phase_dim, lmb, ds)
        masks, _, _ = create_mask(dim=self.dim_out, border=mask_border, rang_size=rang_size)
        self.mask = tf.math.reduce_sum(masks, axis=0)
        self.mask_on = mask_on
        self.layers = layers
        self.normalize = normalize

    # The Angular Spectrum Approach
    def propagation(self, input_image):
        ts = to_complex(tf.sigmoid(self.phase))

        m0 = to_complex(input_image)
        # Forward propagation
        for i in range(len(self.Hs)-1):
            m0_diff = diff(m0, self.Hs[i], dim_out=self.phase_dim)
            m0 = tf.math.multiply(m0_diff, ts[i])
        ml = diff(m0, self.Hs[-1], dim_out=self.dim_out)

        # output
        m_out = tf.math.square(tf.math.abs(ml))
        if self.mask_on:
            m_out = m_out * self.mask
        if self.normalize:
            m_out = m_out / tf.reduce_max(m_out)
        return m_out

    @tf.function
    def __call__(self, input_image):
        return tf.vectorized_map(fn=self.propagation, elems=input_image, fallback_to_while_loop=True)


class D2NNOutput(DNN):
    def __init__(self, layers, dim_in, dim_out, phase_dim, lmb, ds, amp_or_phase):
        super(D2NNOutput, self).__init__(layers, dim_in, dim_out, phase_dim, lmb, ds, amp_or_phase)

    # The Angular Spectrum Approach
    def propagation(self, input_image, phase_input):
        ts = []
        for i, a_or_p in enumerate(self.amp_or_phase):
            if a_or_p == 'amp':
                ts.append(to_complex(tf.sigmoid(phase_input[i])))
            elif a_or_p == 'phase':
                ts.append(tf.math.exp(tf.math.multiply(to_complex(-2j * pi), to_complex(tf.sigmoid(phase_input[i])))))

        m0 = to_complex(input_image)
        # Forward propagation
        for i in range(len(self.Hs)-1):
            m0_diff = diff(m0, self.Hs[i], dim_out=self.phase_dim)
            m0 = tf.math.multiply(m0_diff, ts[i])
        ml = diff(m0, self.Hs[-1], dim_out=self.dim_out)

        # output
        m_out = tf.math.square(tf.math.abs(ml))
        return m_out

    def __call__(self, input_image, phase_input):
        return self.propagation(input_image, phase_input)


