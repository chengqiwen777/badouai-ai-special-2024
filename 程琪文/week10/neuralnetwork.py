import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_nodes = input_size
        self.hidden_nodes = hidden_size
        self.output_nodes = output_size

        self.learning_rate = learning_rate

        self.Wih = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.Who = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T


        hidden_inputs = np.dot(self.Wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.Who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.Who.T, output_errors * final_outputs * (1 - final_outputs))

        self.Who += self.learning_rate * np.dot((output_errors * final_outputs * (1-final_outputs)),
                                                np.transpose(hidden_outputs))
        self.Wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)),
                                                np.transpose(inputs))



    def query(self, inputs):
        hidden_inputs = np.dot(self.Wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.Who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        print(final_outputs)

        return final_outputs

input_size = 784
hidden_size = 100
output_size = 10
learning_rate = 0.1
net = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

f = open('./dataset/mnist_train.csv', 'r')
train_lines = f.readlines()
f.close()

epochs = 5
for e in range(epochs):
    for l in train_lines:
        all_values = l.split(',')
        input_array = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01

        target = np.zeros(10) + 0.01
        target[int(all_values[0])] = 0.99
        net.train(input_array, target)

f = open('./dataset/mnist_test.csv', 'r')
test_lines = f.readlines()
f.close()
scores = []

for record in test_lines:
    all_values = record.split(',')
    correct_num = int(all_values[0])
    print('输入的数字是:', correct_num)
    input_array = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    output = net.query(input_array)

    label = np.argmax(output)
    print('网络推测的数字是:', label)
    if label == correct_num:
        scores.append(1)
    else:
        scores.append(0)

print(scores)
scores_array = np.array(scores)
print('Accuracy: ', scores_array.sum() / scores_array.size)

