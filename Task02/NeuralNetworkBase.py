import numpy as np

class MLP:
    def __init__(self, layers, alpha, activation='sigmoid', bias=False):
        self.Weights = []  # weights for each layer
        self.layers = layers  # network architecture (list of layer sizes)
        self.alpha = alpha  # learning rate
        self.bias = bias  # whether to include bias
        self.activation = activation  # activation function name
        
        # Initialize the weights
        for i in range(0, len(layers) - 1):
            # Add 1 to the input layer size if bias is True
            input_size = layers[i] + (1 if self.bias else 0)
            output_size = layers[i + 1]
            print("Input size for layer", i + 1, ":", input_size)
            print("Output size: from layer", i + 1, ":", output_size)
            # Generate weights with normalization
            w = np.random.randn(input_size, output_size) / np.sqrt(layers[i])
            self.Weights.append(w)
    
    # Define the sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Define the derivative of the sigmoid activation function
    def derivative_sigmoid(self, x):
        return x * (1 - x)
    
    def tanh(x):
        return np.tanh(x)

    def derivative_tanh(x):
        return 1 - np.tanh(x)**2    
    
    def __str__(self):
        net_info = f"Neural Network Structure: {self.layers}\n"
        net_info += f"Learning Rate: {self.alpha}\n"
        net_info += f"Bias Enabled: {self.bias}\n"
        net_info += f"Weights:\n"
        for i, weight in enumerate(self.Weights):
            net_info += f" Layer {i} -> {i+1}:\n{weight}\n"
        return net_info


# Example usage
# 4 inputs, 3 hidden layers with 6, 5, and 4 neurons respectively, and 3 outputs
layers = [4, 6, 5, 4, 3]
alpha = 0.01
bias = True

nn = MLP(layers, alpha, activation='sigmoid', bias=bias)
print(nn)
