#ANN for the Gaussian plume model

import gaussian_plume_model as gp
import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as activation_function
import random


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
    


def let_ann_play(prob,sensor,wind,output, train_percent):
    prob_no_sensor = prob #Probability that a sensor is turned off
    sensor_noise = sensor #Noise amount for input data
    wind_noise = wind #Noise amount for wind data collection
    output_noise = output #Noise for output
    
    #Read in the data
    print('Reading in data...')
    concentrations, wind_dir, leak_loc = gp.read_data('data_XL_new.txt')

    #Let's shuffle the data a bit
    combined = list(zip(concentrations, wind_dir, leak_loc))
    random.shuffle(combined)
    concentrations, wind_dir, leak_loc = zip(*combined)
    
    num_data = len(concentrations)
    #Option for adding a bias node.  Maybe move into NeuralNetwork class in the future
    is_biased = True
    if is_biased:
        num_inputs = len(concentrations[0])**2 + 2
    else:
        num_inputs = len(concentrations[0])**2 + 1

    #Some options for our network
    num_outputs = 2
    learning_rate = 0.1

    #Setup lists for input and output data.
    input_data = []
    output_data = []
    normalized_output = []




    print('Processing concentration data...')
    #Go through and normalize concentrations between 0 and 1 then add some noise to simulate sensor inaccuracy.
    for i in range(num_data):
        c = concentrations[i]
        maximum = max(max(c))
        for j in range(len(c)):
            for k in range(len(c)):
                if random.random() < prob_no_sensor:
                    concentrations[i][j][k] = 0
                else:
                    #print('This sensor is on')
                    concentrations[i][j][k] = float(concentrations[i][j][k])/maximum+ np.random.normal(loc = 0.0, scale = sensor_noise) #Normalize concentrations
                

    
    print('Format other data for Ann to eat...')

    #Prepare take the normalized data and get it ready for ANN to look at.
    for k in range(num_data):
        test_data = []
        for i in range(len(concentrations[k])):
            for j in range(len(concentrations[k])):
                test_data.append(concentrations[k][i][j])

        test_data.append(((wind_dir[k]/360.0)+ np.random.normal(loc = 0.0, scale = wind_noise))%1) #Normalize to make things [0,1]
        if is_biased:
            test_data.append(1)
        input_data.append(test_data)
        output_data.append(leak_loc[k])

    #Scale output between 0 and 1 and add some noise.
    for i in range(len(output_data)):
        normalized_output.append([(1.0/len(concentrations[0]) * output_data[i][0])+np.random.normal(loc = 0.0, scale = output_noise), (1.0/len(concentrations[0]) * output_data[i][1])+np.random.normal(loc = 0.0, scale = output_noise)])

    #Setup our ANN
    simple_network = NeuralNetwork(no_of_in_nodes = num_inputs, no_of_out_nodes = num_outputs, no_of_hidden_nodes = num_inputs, learning_rate = 0.1)

    print('Train Ann, train...')
    #Train ANN
    size_of_training_data = int(num_data * train_percent)
    size_of_testing_data = num_data - size_of_training_data

    for i in range(size_of_training_data):
        simple_network.train(input_data[i], normalized_output[i])    
    
    print('Evaluate Ann...')
    #See what ANN learned.
    print('Scenario Information:')
    print('Noise in sensors: %.2f %%'%(sensor_noise*100))
    print('Noise in wind measurement: %.2f %%'%(wind_noise*100))
    print('Noise in output data: %.2f %%'%(output_noise*100))
    print('Average number of sensors that are on: %d / %d'%(int((1-prob_no_sensor)*len(concentrations[0])**2), len(concentrations[0])**2))

    max_error = -100
    min_error = 100
    
    for tol in np.arange(0,.5,0.05):
        num_correct = 0
        total_error = 0
        missed = 0
        for i in range(size_of_testing_data):
            xGuess, yGuess = simple_network.run(input_data[-i])
            error = ((xGuess - normalized_output[-i][0])**2+(yGuess-normalized_output[-i][1])**2)**0.5

            if(error > max_error):
                max_error = error

            if(error < min_error):
                min_error = error
            
            if(error[0] < tol):
                num_correct = num_correct + 1
            else:
                total_error += error
                missed += 1
        if missed == 0:
            average_error = 0
        else:
            average_error = total_error/missed
        percent_correct = 100.0*num_correct/size_of_testing_data
        print('ANN got %.2f%% of guesses right within %.0f%% of the width of the sensored area'%(percent_correct,tol*100))
        print('ANN\'s average distance out of tolerance was %.2f units for guesses within %.2f%% of the width of the sensored area\n'%(average_error,tol*100))
    print('Maximum error: %.2f'%(max_error))
    print('Minimum error: %.2f'%(min_error))

    return simple_network,input_data,normalized_output

def sample_points(nn,input_data,output_data,num_samples):
    sample_indices = random.sample(range(len(input_data)),num_samples)
    sample = []

    for i in sample_indices:
        sample.append([nn.run(input_data[i]), output_data[i]])

    return sample

"""
Example:
The following two lines will create a neural network called nn and take 10 samples.

nn,input_data,output_data = let_ann_play(.8,.15,.05,.05,0.9)
samples = sample_points(nn,input_data,output_data,10)
"""

#for p in np.arange(0.0,0.95,0.05):
#    let_ann_play(p,0.2,0.05,0.05)

#for n in np.arange(0.05,0.5,0.05):
#    let_ann_play(0.8,n,0.05,0.05)
