import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import MinMaxScaler from sklearn
from sklearn.preprocessing import MinMaxScaler
# Import floor function from math module
from math import floor
from sklearn.metrics import accuracy_score

from extract_ssi_data import *


class LSTM(nn.Module):
    """LSTM time series prediction model

    Attributes:
        num_classes (int): Size of output sample for nn.Linear
        input_size (int): Number of features fed to the model
        hidden_size (int): Number of neurons in each layer
        num_layers (int): Number of layers in the network
        fc: Instance of the nn.Linear module
        lstm: Instance of the LSTM module

    """

    def __init__(self, seq_length, num_classes=1, input_size=1, hidden_size=1, num_layers=1):
        """ Initialize LSTM object

        Args:
            seq_length (int): Sequence length for the input
            num_classes (int): Size of output sample for nn.Linear
            input_size (int): Number of features fed to the model. Defaults to 1
            hidden_size (int): Number of neurons in each layer. Defaults to 1
            num_layers (int): Number of layers in the network. Defaults to 1

        """
        super(LSTM, self).__init__()

        # Set the class attributes
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # Define the lstm model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # Define instance of nn.Linear
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """  Propagate through the NN network layers

        Args:
        x (torch):  is the input features
        """
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device)

        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

def create_sequences(data, seq_length):
    """ Create sequences from the data

    Params:
        data (list): The data you want to split in sequences
        seq_length (int): the length of the sequences

    Returns:
        A list of features and a list of targets
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        # Create a sequence of features
        x = data[i:(i + seq_length)]
        # Take the next value as a target
        y = data[i + seq_length]
        # Append to lists
        xs.append(x)
        ys.append(y)
    # Convert to ndarrays and return
    return np.array(xs), np.array(ys)


print("GPU Driver is installed: "+str(torch.cuda.is_available()))

device = torch.device('cpu')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

    ### Variables ###

# If true the test results and graphs will be printed to a folder. 
# If False plots will be opened in a new window
print_tests_to_folder = False

# If true it will loop through a range of hidden_size and learning_rate
loop_through_tests = False

# Test folder name. The folder we want to print the tests to
test_folder = "test"

# Percentage of test size
test_size_pct = 0.20

# Sequence length
seq_length = 14

# Number of iterations
num_epochs = 100

# Print each 10th epoch value
epoch_print_interval = 100

# Learning rate and hidde size, will only be used if loop_through_tests is False
learning_rate = 0.4
hidden_size = 6

# Other variables
input_size = 1
num_layers = 1
num_classes = 1


if loop_through_tests:
    # Testing range for leaning rate
    learning_rate_range = np.arange(0.1, 0.6, 0.1)
    # Testing range for hidden size
    hidden_size_range = np.arange(1,7,1)
else:
    learning_rate_range = [learning_rate]
    hidden_size_range = [hidden_size]


# Initialize counter
i = 1




# Load flight data from seaborn library
df = extract_ssi_data()
# Convert monthly passengers to float
df = df.values.astype(float)

# Define a scaler to normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
# Scale data. Data is fit in the range [-1,1]
data_normalized = scaler.fit_transform(df.reshape(-1, 1))

# Create feature sequences and targets
x, y = create_sequences(data_normalized, seq_length)

# Split data in training and test
train_size = int(floor(len(df)*(1-test_size_pct)))

# Convert all feature sequences and targets to tensors
x_data = torch.Tensor(np.array(x)).to(device)
y_data = torch.Tensor(np.array(y)).to(device)

# Split train and test data and convert to tensors
x_train = torch.Tensor(np.array(x[0:train_size])).to(device)
y_train = torch.Tensor(np.array(y[0:train_size])).to(device)

x_test = torch.Tensor(np.array(x[train_size:len(x)])).to(device)
y_test = torch.Tensor(np.array(y[train_size:len(y)])).to(device)


# Loop over our desired ranges
for learning_rate in learning_rate_range:
    for hidden_size in hidden_size_range:
        # Create LSTM object
        lstm = LSTM(seq_length, num_classes, input_size, hidden_size, num_layers)
        # Mean squared error loss function defined
        loss_fn = torch.nn.MSELoss()
        # Adam optimizer is used
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)


        nn_log = {"train_loss": [], "test_loss": []}


        # Train the model
        for epoch in range(num_epochs):
            y_pred = lstm(x_train)
            optimizer.zero_grad()

            # Get loss function
            loss = loss_fn(y_pred, y_train)

            # Make prediciton
            y_test_pred = lstm(x_test)
            # Get loss function
            test_loss = loss_fn(y_test_pred, y_test)

            # test_accuracy = np.mean(y_test.detach().numpy() == p_test)

            # Backward propagate
            loss.backward()

            optimizer.step()

            # Save loss
            nn_log["train_loss"].append(loss.item())
            nn_log["test_loss"].append(test_loss.item())
            # Save accuracy
            # nn_log["accuracy"].append(test_accuracy*100)

            # Print loss
            if epoch % 10 == 0:
                print("Epoch: %d, Train loss: %1.5f, Test loss: %1.5f" % (epoch, loss.item(), test_loss.item()))


        # Plot loss by epochs
        plt.plot(nn_log["train_loss"])
        plt.plot(nn_log['test_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(["Train loss", "Test loss"], loc='upper left')
        plt.grid(color='gray', linestyle='-', linewidth=0.1)

        if print_tests_to_folder:
            plt.savefig("{}/{}_loss.png".format(test_folder, i))
            plt.clf()
        else:
            plt.show()



        # Plot train and predict data
        train_predict = lstm(x_data)
        data_predict = train_predict.data.numpy()
        dataY_plot = y_data.data.numpy()

        data_predict = scaler.inverse_transform(data_predict)
        dataY_plot = scaler.inverse_transform(dataY_plot)

        plt.axvline(x=train_size, c='r', linestyle='--')

        plt.plot(dataY_plot)
        plt.plot(data_predict)
        plt.suptitle('Time-Series Prediction')
        plt.grid(color='gray', linestyle='-', linewidth=0.1)


        if print_tests_to_folder:
            plt.savefig("{}/{}_pred.png".format(test_folder, i))
            plt.clf()
        else:
            plt.show()


        # Calculate mean accuracy 

        y_pred = scaler.inverse_transform(lstm(x_test).detach().numpy())
        y_real = scaler.inverse_transform(y_test.detach().numpy())
        test_accuracy = np.mean((abs(y_real - y_pred) / y_real) * 100)
        print(f'Accuracy: {test_accuracy}')

        if print_tests_to_folder:
            file_object = open('{}/tests.txt'.format(test_folder), 'a')
            file_object.write("Test {}".format(i))
            file_object.write("\n Test size: {}".format(test_size_pct))
            file_object.write("\n Epochs: {}".format(num_epochs))
            file_object.write("\n Fineal prediction accuracy: {}".format(test_accuracy))
            file_object.write("\n Hidden size: {}".format(hidden_size))
            file_object.write("\n Num layers: {}".format(num_layers))
            file_object.write("\n Learning rate: {}".format(learning_rate))
            file_object.write("\n Num classes: {}".format(num_classes))
            file_object.write("\n Input size: {}".format(input_size))
            file_object.write("\n Loss: ")

            for ep in np.arange(0, num_epochs, epoch_print_interval):
                file_object.write("\n   Epoch: {}, Train loss: {}, Test loss: {}".format(ep, nn_log["train_loss"][ep], nn_log["test_loss"][ep]))
            file_object.write("\n \n")
            file_object.close()

        i += 1

        
