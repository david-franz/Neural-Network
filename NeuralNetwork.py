import numpy as np

class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        return (1 / (1 + np.exp(-input)))

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = [] # should be size 2
        for i in range(self.num_hidden):
            weighted_sum = 0
            
            for j in range(self.num_inputs):
                weighted_sum += inputs[j] * self.hidden_layer_weights[j][i]
            
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            weighted_sum = 0
            
            for j in range(self.num_hidden):
                weighted_sum += hidden_layer_outputs[j] * self.output_layer_weights[j][i]
            
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        output_layer_betas = np.zeros(self.num_outputs)
        for i in range(len(output_layer_outputs)):
            if i == desired_outputs[0]:
                output_layer_betas[i] = 1 - output_layer_outputs[i]
            else:
                output_layer_betas[i] = 0 - output_layer_outputs[i]
        # print('OL betas: ', output_layer_betas)


        hidden_layer_betas = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            beta = 0.
            for j in range(self.num_outputs-1): # check I need the -1
                beta += (self.output_layer_weights[j][i] * output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j])
            hidden_layer_betas[i] = beta
        # print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                delta_output_layer_weights[i][j] = self.learning_rate * hidden_layer_outputs[i] * output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j]
        # print(delta_output_layer_weights)

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                delta_hidden_layer_weights[i][j] = self.learning_rate * inputs[i] * hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) * hidden_layer_betas[j]
        # print(delta_hidden_layer_weights)

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                self.output_layer_weights[i][j] += delta_output_layer_weights[i][j]

        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                self.hidden_layer_weights[i][j] += delta_hidden_layer_weights[i][j]

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch+1)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
            
                max_prediction = max(output_layer_outputs[0], max(output_layer_outputs[1], output_layer_outputs[2]))
                predicted_class = None
                for index in range(len(output_layer_outputs)):
                    if output_layer_outputs[index] == max_prediction:
                        predicted_class = index
                        break

                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            # print('Hidden layer weights \n', self.hidden_layer_weights)
            # print('Output layer weights  \n', self.output_layer_weights)

            assert len(predictions) == len(desired_outputs)
            num_correct_predictions = 0
            for i in range(len(predictions)):
                if desired_outputs[i] == predictions[i]:
                    num_correct_predictions += 1

            acc = num_correct_predictions / len(predictions)
            print('acc = ', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            
            max_prediction = max(output_layer_outputs[0], max(output_layer_outputs[1], output_layer_outputs[2]))
            predicted_class = None
            for index in range(len(output_layer_outputs)):
                if output_layer_outputs[index] == max_prediction:
                    predicted_class = index
                    break
            predictions.append(predicted_class)
        return predictions