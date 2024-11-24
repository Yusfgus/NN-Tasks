import numpy as np
from sklearn.metrics import confusion_matrix

class Activations:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2 
    
def multiply_lists(list1, list2):
    return [a * b for a, b in zip(list1, list2)]

class MLP:
    def __init__(self, hiddenLayersSizes, learningRate, inputLayerSize=5, outputLayerSize=3, activation='sigmoid', epochs = 1000 ,bias=True):
        self.inputLayerSize = 5
        self.weights = []  # weights for each layer
        self.hiddenLayersSizes = hiddenLayersSizes  # network architecture (list of layer sizes)
        self.hiddenLayersSizes.insert(0, inputLayerSize)
        self.hiddenLayersSizes.append(outputLayerSize)
        self.learningRate = learningRate  # learning rate
        self.activation = activation  # activation function name
        self.bias = bias  # whether to include bias
        self.activations = [] # list of activation result (f(Net[i])) for each layer
        self.layersNets = [] # contains lists of nets for each layer (ex. [[layer_1_Nets], [layer_2_Nets]], [...]).
        self.sigmasList = []
        self.epochs = epochs # no of epochs to perform
        self.activation = Activations.sigmoid if activation == 'sigmoid' else Activations.tanh if activation == 'tanh' else None
        self.activation_derivative = Activations.sigmoid_derivative if activation == 'sigmoid' else Activations.tanh_derivative if activation == 'tanh' else None

        # Initialize the weights for each layer
        for i in range(len(self.hiddenLayersSizes) - 1):  # Iterate through the layers, excluding the last one
            # currentLayerSize = self.hiddenLayersSizes[i] + (1 if self.bias else 0)
            currentLayerSize = self.hiddenLayersSizes[i] + 1
            nextLayerSize = self.hiddenLayersSizes[i + 1]

            # Randomly initialize weights for this layer
            self.weights.append(np.random.randn(currentLayerSize, nextLayerSize))

        # print("Initial Weights : ", self.weights)

    # Feedforward function
    def feedForward(self, X):
        self.activations = [X]
        self.layersNets = []
        for i in range(len(self.weights)):
            self.activations[i] = np.array(self.activations[i]).reshape(1, -1)
            CurrentLayerNets = np.dot(self.activations[i], self.weights[i])

            self.layersNets.append(CurrentLayerNets)
            
            # # Apply activation function (sigmoid for hidden layers, no activation for output layer)
            CurrentLayerNets = self.activation(CurrentLayerNets)

            if i != len(self.weights) - 1:  # If not the output layer
                # Add bias to the next layer input
                CurrentLayerNets = np.append(CurrentLayerNets, 1 if self.bias else 0)
            
            # Append the result to the activations list
            self.activations.append(CurrentLayerNets)    
 
        return self.activations[-1]  # Return the output of the final layer
    

    def backPropagation(self, X, y):
        # Feedforward
        output_activation = self.feedForward(X)

        self.sigmasList = []
        sigmaOutput = (y - output_activation) * self.activation_derivative(self.layersNets[-1])

        sigmas = sigmaOutput
        for i in range(len(self.weights)-1, 0, -1):
            sigmas = multiply_lists(np.dot(sigmas , self.weights[i][:-1, :].T) , self.activation_derivative(self.layersNets[i-1]))
            self.sigmasList.append(sigmas)

        self.sigmasList.reverse()
        self.sigmasList.append(sigmaOutput)

        self.ModifyWeights()


    def ModifyWeights(self):
        for i, sigmas in enumerate(self.sigmasList):
            learningRateList = [self.learningRate] * np.array(sigmas).shape[1]

            newWeights = np.array(np.array(np.dot(np.array(multiply_lists(learningRateList, sigmas)).T, self.activations[i]).T))
            oldWeights = np.array(self.weights[i])
            self.weights[i] =  newWeights + oldWeights
        
    def calculate_accuracy(self, X_train, y_train):
        correct_train = 0
        total_train = len(X_train)

        # Predictions on the training data
        for i in range(total_train):
            predicted = self.predict(X_train.iloc[i])

            actual = y_train.iloc[i]
            if predicted == actual:
                correct_train += 1

        # Calculate accuracy
        train_accuracy = correct_train / total_train * 100
        return train_accuracy
    
    def fit(self, X, y):
        best_weights = None
        least_total_loss = float('inf')

        X = X.assign(bias= 1 if self.bias else 0)
        for epoch in range(self.epochs):
            total_loss = 0
            # Loop over each training sample
            for i in range(X.shape[0]):
                X_sample = X.iloc[i].values.reshape(1, -1)  # Convert row to numpy array and reshape it
                y_sample = y.iloc[i].reshape(1, -1)  # Convert y sample to scalar if it's a pandas Series and reshape it

                self.backPropagation(X_sample, y_sample)

                # Calculate loss (Mean Squared Error) for monitoring
                output = self.activations[-1]
                total_loss += np.mean((y_sample - output) ** 2)
            
            print(f"\rcompleted {((epoch+1)/self.epochs)*100:.0f}%", end="")
            if total_loss < least_total_loss:
                least_total_loss = total_loss
                best_weights = self.weights

        self.weights = best_weights
        print("Best weights: ", best_weights)

    def predict(self, X):
        X_copy = X.copy()

        if len(X_copy) == self.inputLayerSize: # does not have a bias 
            biasEnable = 1 if self.bias else 0
            X_copy = np.append(X_copy, biasEnable)

        # Perform a forward pass
        output = self.feedForward(X_copy)

        # Predicted class: index of the highest output neuron
        predicted_class = np.argmax(output)
        return predicted_class
    
    def calculate_accuracy_and_confusion_matrix(self, X_train, y_train, X_test, y_test):
        # Initialize counters for accuracy and confusion matrix
        correct_train = 0
        correct_test = 0
        total_train = len(X_train)
        total_test = len(X_test)
        
        # Initialize confusion matrices
        cm_train = np.zeros((3, 3), dtype=int)  # For 3 classes
        cm_test = np.zeros((3, 3), dtype=int)

        # Predictions on the training data
        for i in range(total_train):
            predicted = self.predict(X_train.iloc[i])
            actual = y_train.iloc[i]
            if predicted == actual: correct_train += 1
            cm_train[actual][predicted] += 1

        # Predictions on the testing data
        for i in range(total_test):
            predicted = self.predict(X_test.iloc[i])
            actual = y_test.iloc[i]
            if predicted == actual: correct_test += 1
            cm_test[actual][predicted] += 1

        # Calculate accuracy
        train_accuracy = correct_train / total_train * 100
        test_accuracy = correct_test / total_test * 100

        # Print Results
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Testing Accuracy: {test_accuracy:.2f}%")
        print("Training Confusion Matrix:")
        print(cm_train)
        print("Testing Confusion Matrix:")
        print(cm_test)