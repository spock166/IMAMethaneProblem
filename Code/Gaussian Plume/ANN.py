#ANN for the Gaussian plume model

import gaussian_plume_model as gp
import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as activation_function


#We'll use this function to generate the initial weights for our ANN
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#Class borrowed from https://www.python-course.eu/neural_networks_with_python_numpy.php
class NeuralNetwork:
    
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes 
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate  
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                       self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                        self.no_of_hidden_nodes))
             
    
    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)
        
        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)     
        tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector.T)
    
    def run(self, input_vector):
        """
        running the network with an input vector input_vector. 
        input_vector can be tuple, list or ndarray
        """
        
        # turning the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector



#Read in the data
concentrations, wind_dir, leak_loc = gp.read_data('data_medium.txt')

num_data = len(concentrations)

    
for i in range(num_data):
    c = concentrations[i]
    maximum = max(max(c))
    #print(maximum)
    for j in range(len(c)):
        for k in range(len(c)):
            concentrations[i][j][k] = float(concentrations[i][j][k])/maximum+ np.random.normal(loc = 0.0, scale = 0.05) #Normalize concentrations

#print(num_data)
num_inputs = len(concentrations[0])**2 + 1
num_outputs = 2
learning_rate = 0.1

input_data = []
output_data = []
normalized_output = []

#Setup our ANN
simple_network = NeuralNetwork(no_of_in_nodes = num_inputs, no_of_out_nodes = num_outputs, no_of_hidden_nodes = num_inputs, learning_rate = 0.1)


#Prepare data for ANN
input_data = []
output_data = []

for k in range(num_data):
    test_data = []
    for i in range(len(concentrations[k])):
        for j in range(len(concentrations[k])):
            test_data.append(concentrations[k][i][j])

    test_data.append(((wind_dir[k]/360.0)+ np.random.normal(loc = 0.0, scale = 0.01))%1) #Normalize to make things [0,1]
    input_data.append(test_data)
    output_data.append(leak_loc[k])

#Scale output
for i in range(len(output_data)):
    normalized_output.append([(1.0/len(concentrations[0]) * output_data[i][0])+np.random.normal(loc = 0.0, scale = 0.05), (1.0/len(concentrations[0]) * output_data[i][1])+np.random.normal(loc = 0.0, scale = 0.05)])

#Train ANN
size_of_training_data = int(num_data * 0.9)
size_of_testing_data = num_data - size_of_training_data

for i in range(size_of_training_data):
    simple_network.train(input_data[i], normalized_output[i])

#See what ANN learned.  Should add tolerance for correctness \varepsilon <= 0.05??
num_correct = 0
for i in range(size_of_testing_data):
    xGuess, yGuess = simple_network.run(input_data[-i])
    error = ((xGuess - normalized_output[-i][0])**2+(yGuess-normalized_output[-i][1])**2)**0.5
    if(error[0] < 0.1):
        num_correct = num_correct + 1

percent_correct = 100.0*num_correct/size_of_testing_data
print('ANN got ' + str(percent_correct) + '% of guesses right within 10% of the width of the sensored area')
