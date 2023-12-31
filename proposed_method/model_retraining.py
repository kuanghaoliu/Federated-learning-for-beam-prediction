import tensorflow as tf
import numpy as np
from BeamPredNet import Model
import scipy.io as sio

dataset_maxQ = 9
Q = 9
m = 9
total = (m+1)*(dataset_maxQ+1)
Q_array = list(range(0, (Q+2)*10, 10))
lr = 5e-5


def Compute_Gn(y_pred, beam_power, y_true):
    BL = np.zeros((Q+1, m))
    p = 0
    for i in range(len(y_pred)):
        a = 0
        for j in range(Q+1):
            for k in range(m):
                train_ans = np.squeeze(y_pred[i, a, :])
                # find the index with the maximum probability
                train_index = np.argmax(train_ans)
                # counting normalized beamforming gain
                BL[j, k] = BL[j, k] + (beam_power[i, a, train_index] / max(
                beam_power[i, a, :])) ** 2
                if train_index == y_true[i, a]:
                    p = p+1
                a = a + 1
    BL = BL / len(y_pred)
    ACC = p / len(y_pred) / ((Q+1)*m)
    print(np.mean(BL, axis=0))
    return BL, ACC

def loss_fn(y_true, y_pred):
    #y_true = y_true[:, :, 0]
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    output = cce(y_true, y_pred) * ((Q+1)*m)
    return output


class Gain(tf.keras.metrics.Metric):
    def __init__(self, name='Gain', **kwargs):
        super(Gain, self).__init__(name=name, **kwargs)
        self.Gain = self.add_weight(name='Gain', initializer='zeros')
        self.acc = self.add_weight(name='acc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        bp = y_true[:, :, 1:]
        train_index = tf.math.argmax(y_pred, axis=2, output_type=tf.dtypes.int32)
        best_beam = tf.reduce_max(bp, axis=2)
        batch_indices = tf.tile(tf.expand_dims(tf.range(tf.shape(bp)[0]), axis=1), [1, tf.shape(bp)[1]])
        row_indices = tf.tile(tf.expand_dims(tf.range(tf.shape(bp)[1]), axis=0), [tf.shape(bp)[0], 1])
        gather_indices = tf.stack([batch_indices, row_indices, train_index], axis=2)
        train_beam = tf.gather_nd(bp, gather_indices)
        values = tf.square(train_beam / best_beam)
        values = tf.reduce_mean(values)
        self.Gain.assign(values)

    def result(self):
        return self.Gain


class Acc(tf.keras.metrics.Metric):
    def __init__(self, name='Acc', **kwargs):
        super(Acc, self).__init__(name=name, **kwargs)
        self.Acc = self.add_weight(name='Acc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.dtypes.int32)
        values = tf.math.equal(y_true, tf.math.argmax(y_pred, axis=2, output_type=tf.dtypes.int32))
        values = tf.cast(values, self.dtype)
        values = tf.reduce_sum(values) / len(y_pred) / 90
        self.Acc.assign(values)

    def result(self):
        return self.Acc

def load_data(path):
    data = sio.loadmat(path)
    channels = data['MM_data']
    channel_train = channels[:, :, 0: total: (m + 1), :]
    channel_train = np.transpose(channel_train, (0, 2, 3, 1))
    channel_train = np.delete(channel_train, slice(0, dataset_maxQ-Q), axis=1)
    beam_power = data['beam_power']
    beam_power = np.delete(beam_power, slice(0, (dataset_maxQ-Q)*10), axis=1)
    beam_power = np.delete(beam_power, Q_array, axis=1)
    labels = data['beam_label'] - 1
    labels = np.delete(labels, slice(0, (dataset_maxQ-Q)*10), axis=1)
    labels = np.delete(labels, Q_array, axis=1)
    return channel_train, labels, beam_power


for R in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
    x, y, bp_train = load_data('file_path')
    model = Model(timespans=Q+1, pred_points=m)
    model.build((32, Q+1, 64, 2))
    model.load_weights('file_path')

    BL_out = []
    acc_test = []

    predictions = model.predict(x[-128:], verbose=0)
    BL, ACC = Compute_Gn(predictions, bp_train[-128:], y[-128:])
    print('Test Accuracy: {}'.format(ACC))
    print('Beam Power Gain: {}'.format(np.mean(BL)))
    BL_out.append(BL)
    acc_test.append(ACC)

    x_train = x[256:-128]
    y_train = y[256:-128]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999),
                      loss=loss_fn,
                      metrics=tf.keras.metrics.SparseCategoricalAccuracy())

    history = model.fit(x_train, y_train, epochs=20, batch_size=32)

    predictions = model.predict(x[-128:], verbose=0)
    BL, ACC = Compute_Gn(predictions, bp_train[-128:], y[-128:])
    print('Test Accuracy: {}'.format(ACC))
    print('Beam Power Gain: {}'.format(np.mean(BL)))
    BL_out.append(BL)
    acc_test.append(ACC)

    print('------------------------------------------------------------------')
    print(np.shape(BL_out))
    print(acc_test)
    mat_name = './result/Federated/retraining_rand_'+str(R)+'.mat'
    sio.savemat(mat_name, {'acur_test': acc_test, 'BL_test': BL_out})
