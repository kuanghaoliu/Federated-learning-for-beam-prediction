import numpy as np
import tensorflow as tf


# building model with ODE-LSTM cell
class Model(tf.keras.Model):
    # model initialization
    def __init__(
        self,
        in_features=256,
        hidden_size=256,
        out_feature=64,
        return_sequences=True,
        solver_type="fixed_euler",
        batch_size=32,
        timespans=10,
        pred_points=9
    ):
        super(Model, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences
        self.batch_size = batch_size
        self.timespans = timespans
        self.pred_points = pred_points

        # CNN preprocessing
        #self.bn0 = tf.keras.layers.BatchNormalization(axis=3)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(1, 3), strides=(1, 3), padding='same')
        #self.bn1 = tf.keras.layers.BatchNormalization(axis=3)
        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=(1, 3), strides=(1, 3), padding='same')
        #self.bn2 = tf.keras.layers.BatchNormalization(axis=3)
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(1, 3), strides=(1, 3), padding='same')
        #self.bn3 = tf.keras.layers.BatchNormalization(axis=3)

        # LSTM layer for input learning
        self.lstm_cell = tf.keras.layers.LSTMCell(256)

        # FC for output
        self.drop = tf.keras.layers.Dropout(0.3)
        self.fc = tf.keras.layers.Dense(self.out_feature)
        self.relu = tf.keras.layers.ReLU()
        self.permute = tf.keras.layers.Permute((1, 3, 2))

    # timespans: number of beam trainings
    # pre_points: number of predicted instants between two times of beam training
    def call(self, x):
        batch_size = self.batch_size
        seq_len = self.timespans

        outputs = []
        last_output = tf.zeros((batch_size, self.out_feature))

        # CNN preprocessing
        #x = self.bn0(x)
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        x = self.conv1(x)
        #print(x.shape)
        #x = self.bn1(x)
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        x = self.relu(x)
        x = self.conv2(x)
        #print(x.shape)
        #x = self.bn2(x)
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        x = self.relu(x)
        x = self.conv3(x)
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        #print(x.shape)
        #x = self.bn3(x)
        x = self.relu(x)

        P_dim_size = x.shape[3]
        x = tf.keras.layers.AvgPool2D(pool_size=(1, P_dim_size), padding='same')(x)
        x = self.permute(x)
        x = tf.squeeze(x)

        # define variables of LSTM learning
        new_h = tf.zeros((batch_size, self.hidden_size))
        new_c = tf.zeros((batch_size, self.hidden_size))
        pre_points = self.pred_points

        # ODE-LSTM learning
        for t in range(seq_len):
            inputs = x[:, t]
            current_output = []

            for num in range(pre_points):
                y1, (new_h, new_c) = self.lstm_cell(inputs, [new_h, new_c])
                y1 = self.drop(y1)
                y1 = self.fc(y1)
                y1 = tf.nn.softmax(y1)
                current_output.append(y1)

            outputs.append(current_output)

            # outputs = tf.stack(outputs, axis=1)  # return entire sequence
        outputs = tf.convert_to_tensor(outputs, tf.float32)
        #print(outputs.shape)
        output_tensor = []

        d_sery = np.linspace(np.float32(1) / pre_points / 2, np.float32(1) - np.float32(1) / pre_points / 2,
                             num=pre_points)
        # for all predictions after 10 beam trainings
        # output_tensor = tf.zeros((batch_size, 9*seq_len, self.out_feature))
        for loss_count in range(seq_len):
            # time offset between two times of beam training
            for d_count in range(pre_points):
                t_d = np.abs(d_count * (1 / (pre_points + 1)) + (1 / (pre_points + 1)) - d_sery)
                # min_location is the closest time stamp
                min_location = np.argmin(t_d)
                # calculate prediction loss
                output_tensor.append(tf.squeeze(outputs[loss_count, d_count, :, :]))
        output_tensor = tf.convert_to_tensor(output_tensor, tf.float32)
        output_tensor = tf.transpose(output_tensor, perm=[1, 0, 2])
        return output_tensor