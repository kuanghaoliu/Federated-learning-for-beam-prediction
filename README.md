This repo contains the source codes for reproducing the results in the following paper

Cheng-Jui Chuang and Kuang-Hao (Stanley) Liu, 

Please direct any questions to irat@ee.nthu.edu.tw. We will be happy to discuss and provide more details.

# Generate training data
Here are the steps for generating training data. The required files can be found in the folder *data*.  
_The following datasets are generated with reference to "[Continuous-Time mmWave Beam Prediction With ODE-LSTM Learning Architecture](https://ieeexplore.ieee.org/document/9944873)"._
   1. Download DeepMIMO and the data files of Raytracing scenario O1 in 28 GHz operating frequency from the DeepMIMO website : https://www.deepmimo.net/
   2. Set the simulation in *parameters.m*.
   3. Generate the millimeter wave channel using *DeepMIMO_Dataset_Generator.m*.
   4. Generate User trajectory at the normalized prediction instant $\tau$ with four settings.
      - *generator_fixed_velocity.m*: Generate training, validation, and testing datasets with $v_k=5, 10, 15, 20, 25, 30$ respectively.
      - *generator_constant_velocity.m*: Generate training and validation datasets, where $v_k~Uniform(0,30)$ and each UE $k$ moves within its respective activity range.
      - *generator_random_velocity.m*: Generate training and validation datasets, where each UE $k$ moves randomly within its respective activity range with an average velocity $\overline{v_k}=(k \mod 6)*5+5$.
      - *generator_testing_dataset.m*: Generate a testing dataset for testing models trained with 'constant_velocity_dataset' and 'random_velocity_dataset', where the UE moves at random speeds throughout the full activity range.
      - *generator_retraining_dataset.m*: Generate datasets for the re-training method proposed in the paper.

# Proposed method
See folder *proposed_method*

This is the case where beam tracking (using a small number of probing beams) is performed every $T$ seconds, where $T=100$ ms by default. Each trajectory lasts for 1 second.
- BeamPredNet.py: neural network architecture for BeamPredNet.
- fl_training.py: the main program for training BeamPredNet using federated learning (the main methods proposed in the paper).
- model_retraining.py: the program for re-training the FL-trained model locally.

# Benchmark schemes
The folder *benchmark* contains the files for implementing EKF, Conventional LSTM, Cascaded LSTM, and ODE-LSTM.

# Simulation setup

## Simulation parameters
| Parameter                            | Value  |
|--------------------------------------|--------|
| Carrier frequency $f_c$              | 28 GHz |
| Number of BS antennas $N_{BS}$       | 64     |
| Number of BS codewords $I$           | 64     |
| Number of UE antennas                | 1      |
| Number of UEs $K$                    | 36     |
| Number of previous beam training $Q$ | 9      |
| Noise factor $N_F$                   | 5 dB   |
| BS transmit power                    | 30 dBm |
| Beam training period $T$             | 160 ms |

## Mobility model
We sample different UE locations from the DeeMIMO dataset (https://www.deepmimo.net/) to generate the UE movement trajectory. 
The considered area is a rectangular region that spans a length from rows 1 to 999 and a width covering columns from 1 to 181. 
Furthermore, each UE only moves within its specific activity range.

| Parameter                     | Value                      |  
|-------------------------------|----------------------------|
| BS index                      | 1                          |
| Location range of user        | Row 1~999                  |
| Acceleration                  | $0.2v~\text{m}/\text{s}^2$ |
| Moving direction              | Uniform($0,2\pi$)          |


# Training BeamPredNet with federated learning
There are four neural network components in the proposed BeamPredNet.
- Two residual blocks.
- Lstm cell with 256 hidden states.
- ODESolver with the Euler method as an integrator.
- The final FC layer with a dropout of 0.3.

Use Adam as the optimizer for training on the local client and SGD as the optimizer on the server.  
The details of hyperparameters are shown below.

| Parameter                    | Value             |
|------------------------------|-------------------|
| Local epochs                 | 10                |
| Global rounds                | 30                |
| Batch size                   | 32                |
| Initial client learning rate | $1\times 10^{-4}$ |
| Initial server learning rate | $1$               |
| Learning rate factor         | 0.5               |
| Minimum client learning rate | $1\times 10^{-7}$ |

# Simulation result
The *results* folder is used to draw simulation results in the paper.

