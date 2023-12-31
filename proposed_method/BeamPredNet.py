import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K


class ResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(ResBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d_w1 = self.add_weight("conv2d_w1", self.kernel_size + (self.filters, self.filters), initializer='glorot_uniform')
        self.conv2d_w2 = self.add_weight("conv2d_w2", self.kernel_size + (self.filters, self.filters), initializer='glorot_uniform')
        self.conv2d_b1 = self.add_weight("conv2d_b1", (self.filters,), initializer='zero')
        self.conv2d_b2 = self.add_weight("conv2d_b2", (self.filters,), initializer='zero')
        super(ResBlock, self).build(input_shape)

    def call(self, x):
        y = K.conv2d(x, self.conv2d_w1, padding="same")
        y = K.bias_add(y, self.conv2d_b1)
        y = K.relu(y)

        y = K.conv2d(y, self.conv2d_w2, padding="same")
        y = K.bias_add(y, self.conv2d_b2)
        y = K.relu(y)

        y = y+x
        return y

class ConcatConv2d(tf.keras.Model):

    def __init__(self, dim_in, dim_out, ksize, stride, padding=0, dilation=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = tf.keras.layers.Conv2DTranspose if transpose else tf.keras.layers.Conv2D

        padding = 'same'if padding == 1 else 'valid'
        self._layer = module(
            dim_out, kernel_size=ksize, strides=stride, padding=padding,
            dilation_rate=dilation,
            use_bias=bias
        )

    def call(self, t, x):
        t = tf.cast(t, tf.float32)
        tt = tf.ones_like(x[:, :, :, :1]) * t  # channel dim = -1
        ttx = tf.concat([tt, x], -1)  # concat at channel dim
        return self._layer(ttx)


class ODEfunc(tf.keras.Model):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        #self.norm0 = norm(dim)
        self.conv1 = ConcatConv2d(dim, dim, (3, 3), (1, 1), 1)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=3)

        #self.norm1 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, (3, 3), (1, 1), 1)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=3)
        #self.norm2 = norm(dim)
        self.conv3 = ConcatConv2d(dim, dim, (3, 3), (1, 1), 1)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=3)
        self.relu = tf.keras.layers.ReLU()

    def call(self, t, x):
        x = tf.cast(x, tf.float32)  # needs an explicit cast
        out = self.conv1(t, x)
        mean_x, std_x = tf.nn.moments(out, axes=[0, 1, 2], keepdims=True)
        out = tf.nn.batch_normalization(out, mean_x, std_x, None, None, 1e-12)
        out = tf.keras.activations.tanh(out)
        out = self.conv2(t, out)

        return out


class ODEBlock(tf.keras.Model):

    def __init__(self, odefunc, **kwargs):
        super(ODEBlock, self).__init__(**kwargs)
        self.odefunc = odefunc
        self.integration_time = tf.convert_to_tensor([0., 1.], dtype=tf.float32)

        self.solver = tfp.math.ode.DormandPrince(
            rtol=1e-04,
            atol=1e-04,
            first_step_size=0.01,
            safety_factor=0.8,
            min_step_size_factor=0.1,
            max_step_size_factor=10.0,
            max_num_steps=None,
            make_adjoint_solver_fn=None,
            validate_args=False,
            name="dormand_prince",
        )

    def call(self, x):
        # self.integration_time = tf.cast(self.integration_time, x.dtype)
        out = self.solver.solve(
                ode_fn=self.odefunc,
                initial_time=0,
                initial_state=x,
                solution_times=self.integration_time,  # tfp.math.ode.ChosenBySolver(elapsed),
            )
        return tf.cast(out.states[1], tf.float32)  # necessary cast


class ODELSTMCell(tf.keras.Model):
    def __init__(self, hidden_size, solver_type):
        super(ODELSTMCell, self).__init__()
        # solver of ordinary differential equation
        self.solver_type = solver_type
        #print("solver_type:", self.solver_type)
        self.fixed_step_solver = solver_type.startswith("fixed_")

        # FC layer of neural ODE
        # fit the derivative
        self.f_node = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='tanh'),
            tf.keras.layers.Dense(hidden_size),
        ])

        # candidate integrand functions
        options = {
            "fixed_euler": self.euler,
            "fixed_heun": self.heun,
            "fixed_rk4": self.rk4,
        }
        if not solver_type in options.keys():
            raise ValueError("Unknown solver type '{:}'".format(solver_type))
        self.node = options[self.solver_type]
        self.drop = tf.keras.layers.Dropout(0.3)
        self.fc = tf.keras.layers.Dense(64)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, new_h, ts):
        # feed the predicted results back
        # new_h is current_state
        # ts is integral duration
        new_h = self.solve_fixed(new_h, ts)
        y = self.drop(new_h)
        y = self.fc(y)
        y = self.softmax(y)
        y = tf.expand_dims(y, axis=0)
        return new_h, y

    def solve_fixed(self, x, ts):
        # integrate the predicted results
        ts = tf.reshape(ts, (-1, 1))
        for i in range(3):  # 3 unfolds (refer to original ODE-LSTM paper)
            # integral step is ts * (1.0 / 3)
            x = self.node(x, ts * (1.0 / 3))
        return x

    # euler, heun and rk4 are different integration methods
    def euler(self, y, delta_t):
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)
        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


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

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(1, 3), strides=(1, 3), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')

        self.ResBlock1 = ResBlock(64, (3, 3))
        self.ResBlock2 = ResBlock(256, (3, 3))

        # define ODE-LSTM cell
        self.rnn_cell = ODELSTMCell(hidden_size, solver_type=solver_type)

        # LSTM layer for input learning
        self.lstm_cell = tf.keras.layers.LSTMCell(256)

        # FC for output
        self.drop = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256)
        self.relu = tf.keras.layers.ReLU()
        self.permute = tf.keras.layers.Permute((2, 1, 3))
        # return outputs
    # timespans: number of beam trainings
    # pre_points: number of predicted instants between two times of beam training
    def call(self, x):
        batch_size = self.batch_size
        seq_len = self.timespans
        pre_points = self.pred_points
        outputs = []

        # ResNet preprocessing
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        x = self.conv1(x)
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        x = self.ResBlock1(x)
        x = self.conv2(x)
        x = self.ResBlock2(x)

        '''
        x = self.conv1(x)
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        x = self.relu(x)
        x = self.conv2(x)
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        x = self.relu(x)
        x = self.conv3(x)
        #x = tf.keras.layers.AvgPool2D(pool_size=(1, 3), padding='same')(x)
        #print('AvgPool', x.shape)
        mean_x, std_x = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean_x, std_x, None, None, 1e-12)
        x = self.relu(x)
        '''
        # define variables of LSTM learning
        new_h = tf.zeros((batch_size, self.hidden_size))
        new_c = tf.zeros((batch_size, self.hidden_size))
        # ODE-LSTM learning
        for t in range(seq_len):
            inputs = x[:, t]

            inputs = self.flatten(inputs)
            inputs = self.fc1(inputs)
            mean_x, std_x = tf.nn.moments(inputs, axes=[0, 1], keepdims=True)
            inputs = tf.nn.batch_normalization(inputs, mean_x, std_x, None, None, 1e-12)
            #inputs = self.relu(inputs)

            # LSTM learning
            out, (new_h, new_c) = self.lstm_cell(inputs, [new_h, new_c])

            # time offset
            ts = (1/(pre_points + 1)) * tf.ones(batch_size)

            # the first point
            new_h1, y1 = self.rnn_cell.call(new_h, ts)

            # other (pre_points - 1) points
            for num in range(pre_points - 1):
                new_h1, y2 = self.rnn_cell.call(new_h1, ts)
                y1 = tf.concat([y1, y2], 0)

            # save the output
            current_output = y1
            outputs.append(current_output)

        outputs = tf.convert_to_tensor(outputs, tf.float32)
        output_tensor = []

        for loss_count in range(seq_len):
            for d_count in range(pre_points):
                output_tensor.append(tf.squeeze(outputs[loss_count, d_count, :, :]))
        output_tensor = tf.convert_to_tensor(output_tensor, tf.float32)
        output_tensor = tf.transpose(output_tensor, perm=[1, 0, 2])

        return output_tensor