import nest_asyncio
nest_asyncio.apply()
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import scipy.io as sio
import timeit
import matplotlib.pyplot as plt
from benchmark.model.model_ODE_LSTM import Model

print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

tf.random.set_seed(0)

NUM_CLIENTS = 36
NUM_EPOCHS = 10
NUM_ROUNDS = 50
BATCH_SIZE = 32
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
HIDDEN_STATE = 256
Beam_Num = 64
m = 9
dataset_maxQ = 9
Q = 9
total = (m+1)*(dataset_maxQ+1)
client_lr = [1e-4]
server_lr = 1
loss_val = []
patience = [0]
Q_array = list(range(0, (Q+2)*10, 10))
print(Q_array)

path_train = 'file_path'
path_valid = 'file_path'
test = 'file_path'
data = sio.loadmat(test)
data_test = data['MM_data']  # beam training received signal
data_test = data_test[:, :, 0: total: (m + 1), :]
data_test = np.transpose(data_test, (0, 2, 3, 1))
data_test = np.delete(data_test, slice(0, dataset_maxQ-Q), axis=1)
#print(data_test.shape)
true_test = data['beam_label'] - 1
true_test = np.delete(true_test, slice(0, (dataset_maxQ-Q)*10), axis=1)
true_test = np.delete(true_test, Q_array, axis=1)
#print(true_test.shape)
beam_power = data['beam_power']
beam_power = np.delete(beam_power, slice(0, (dataset_maxQ-Q)*10), axis=1)
beam_power = np.delete(beam_power, Q_array, axis=1)
#print(beam_power.shape)

def load_dataset(path):
    data = sio.loadmat(bytes.decode(path.numpy()))
    channels = data['MM_data']  # beam training received signal
    channels = channels[:, :, 0: total: (m + 1), :]
    channels = np.transpose(channels, (0, 2, 3, 1))
    channels = np.delete(channels, slice(0, dataset_maxQ-Q), axis=1)
    channel_train = tf.convert_to_tensor(channels, dtype=tf.float32)
    labels = data['beam_label'] - 1
    labels = np.delete(labels, slice(0, (dataset_maxQ-Q)*10), axis=1)
    labels = np.delete(labels, Q_array, axis=1)
    data = collections.OrderedDict((('label', labels), ('data', channel_train)))
    data = tf.data.Dataset.from_tensor_slices(data)
    return data

train_dataset = tff.simulation.datasets.FilePerUserClientData.create_from_dir(path_train, load_dataset)
test_dataset = tff.simulation.datasets.FilePerUserClientData.create_from_dir(path_valid, load_dataset)


def preprocess(dataset):
  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=element['data'],
        y=element['label'])

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]


example_dataset = train_dataset.create_tf_dataset_for_client(
    train_dataset.client_ids[0])
preprocessed_example_dataset = preprocess(example_dataset)

sample_clients = train_dataset.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(train_dataset, sample_clients)

print(f'Number of client datasets: {len(federated_train_data)}')
print(f'First dataset: {federated_train_data[0]}')


class loss_fn(tf.keras.losses.Loss):
    def __init__(self, name='loss', **kwargs):
        super(loss_fn, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        output = cce(y_true, y_pred) * (m*(Q+1))
        return output


def model_fn():
  keras_model = Model(timespans=(Q+1), pred_points=m)
  keras_model.build((BATCH_SIZE, Q+1, 64, 2))
  #print(keras_model.summary())
  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=loss_fn(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      #metrics=[Acc()]
  )


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


def ReduceLR(round):
    if len(loss_val) > 3 and client_lr[-1] != 0.0000001:
        new_lr = client_lr[-1]
        threshold = 0.0001
        if loss_val[-1] > loss_val[-2] * (1-threshold):
            patience.append(patience[-1]+1)
            if patience[-1] > 2:
                new_lr = client_lr[-1] * 0.5
                patience[-1] = 0
                if new_lr < 0.0000001:
                    new_lr = 0.0000001
    else:
        new_lr = client_lr[-1]
    if new_lr != client_lr[-1]:
        print('Client Learning Rate Reduce to:', new_lr)
    return tf.constant(new_lr, tf.float32)


def client_optimizer_fn(learning_rate):
    return tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999)


def main():
    acc = []
    acc_val = []
    acc_test = []
    BL_out = []
    loss_train = []
    loss_test = []

    compressed_mean = tff.aggregators.MeanFactory(
        tff.aggregators.EncodedSumFactory.quantize_above_threshold(
            quantization_bits=8, threshold=20000))

    training_process = tff.learning.algorithms.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn,
        client_learning_rate_fn=ReduceLR,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=server_lr),
        model_aggregator=compressed_mean
        )

    print(training_process.initialize.type_signature.formatted_representation())
    train_state = training_process.initialize()
    federated_test_data = make_federated_data(test_dataset, sample_clients)

    print('number of clients: {}'.format(NUM_CLIENTS))
    print('local epoch: {}, global round: {}'.format(NUM_EPOCHS, NUM_ROUNDS))
    print('local learning rate: {}, global learning rate: {}'.format(client_lr, server_lr))
    evaluation_process = tff.learning.algorithms.build_fed_eval(model_fn)
    evaluation_state = evaluation_process.initialize()
    scce = tf.keras.losses.SparseCategoricalCrossentropy()


    for round_num in range(1, NUM_ROUNDS+1):
        result = training_process.next(train_state, federated_train_data)
        train_state = result.state
        train_metrics = result.metrics
        print('round {:2d}, metrics={}'.format(round_num, train_metrics))
        acc.append(train_metrics['client_work']['train']['sparse_categorical_accuracy'])
        loss_train.append(train_metrics['client_work']['train']['loss'])

        model_weights = training_process.get_model_weights(train_state)
        evaluation_state = evaluation_process.set_model_weights(evaluation_state, model_weights)
        evaluation_output = evaluation_process.next(evaluation_state, federated_test_data)
        print('Eval Accuracy: {}'.format(evaluation_output.metrics['client_work']['eval']['current_round_metrics']['sparse_categorical_accuracy']))
        acc_val.append(evaluation_output.metrics['client_work']['eval']['current_round_metrics']['sparse_categorical_accuracy'])
        loss_val.append(evaluation_output.metrics['client_work']['eval']['current_round_metrics']['loss'])
        client_lr.append(ReduceLR(round_num))

        model_for_inference = Model(timespans=(Q+1), pred_points=m)
        model_for_inference.build((32, Q+1, 64, 2))
        model_for_inference.set_weights(model_weights[0])
        predictions = model_for_inference.predict(data_test, verbose=0)
        BL, ACC= Compute_Gn(predictions, beam_power, true_test)
        BL_out.append(BL)
        acc_test.append(ACC)
        loss_test.append((scce(true_test, predictions).numpy()*(Q+1)*m))
        #m.update_state(true_test, predictions)
        print('Test Accuracy: {}'.format(ACC))
        print('Test Loss: {}'.format(loss_test[-1]))
        print('Beam Power Gain: {}'.format(np.mean(BL)))
        '''
        if round_num == NUM_ROUNDS:
            model_for_inference.save_weights('./result/model_Mix15_E15/')
            print('Model weights Saved')
        '''
        #if round_num % 5==0:
        #    model_for_inference.save_weights('./result/model_Mix'+str(round_num/5)+'/')
        #    print('Model weights Saved')

    plt.plot(acc)
    plt.plot(acc_val)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Global Rounds')
    plt.legend(['Train', 'Test'], loc='upper left')
    filename = 'Rand_Q' + str(Q) + '_C' + str(NUM_CLIENTS) + '_' + str(NUM_EPOCHS) + '_' + str(NUM_ROUNDS) + '_' + str(BATCH_SIZE)
    #plt.savefig(filename)
    plt.show()

    mat_name = './result/Federated/Rand_Q'+ str(Q) + '_C' + str(NUM_CLIENTS) + '_E' + str(NUM_EPOCHS) + '_R' + str(NUM_ROUNDS) + '_B' + str(BATCH_SIZE) + '.mat'
    sio.savemat(mat_name, {'acur_train': acc, 'acur_eval': acc_val, 'acur_test': acc_test,
                           'loss_train': loss_train, 'loss_eval': loss_val, 'loss_test': loss_test,
                           'BL_test': BL_out})
    '''
    data = sio.loadmat(mat_name)
    print(data['acur_train'].shape, data['acur_eval'].shape, data['acur_test'].shape,
          data['loss_train'].shape, data['loss_eval'].shape, data['loss_test'].shape,
          data['BL_test'].shape, )

    new_model = Model(timespans=Q+1, pred_points=m)
    new_model.build((32, 10, 64, 2))
    new_model.load_weights('./result/model/')
    predictions = new_model.predict(data_test, verbose=0)
    BL, ACC = Compute_Gn(predictions, beam_power, true_test)
    print('Test Accuracy: {}'.format(ACC))
    print('Beam Power Gain: {}'.format(np.mean(BL)))
    '''


if __name__ == '__main__':
    execution_time = timeit.timeit(main, number=1)
    days, remainder = divmod(int(execution_time), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, remainder = divmod(remainder, 60)
    print('Execution Time: {}days {}hours {}minutes'.format(days, hours, minutes))
