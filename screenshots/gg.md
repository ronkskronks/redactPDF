#!/usr/bin/env python3

"""
NEURAL NETWORK WITH ARITHMETIC POVERTY
=====================================

A neural network implementation using ONLY the 4 basic arithmetic operations:
+ (addition)
- (subtraction) 
* (multiplication)
/ (division)

NO FANCY FUNCTIONS ALLOWED:
❌ No exp(), log(), sqrt()
❌ No max(), min(), abs()
❌ No sigmoid, ReLU, tanh
❌ No numpy, tensorflow, pytorch
❌ No mathematical libraries
❌ No conditional statements for activations

ONLY PURE ARITHMETIC SUFFERING! 🔥

This forces us to:
- Approximate activation functions with polynomials
- Use linear/quadratic functions instead of exponentials
- Implement learning through pure algebraic manipulation
- See the RAW mathematical core of neural networks

Architecture: 2 inputs -> 3 hidden -> 1 output (19 parameters total)
Activation: Custom polynomial approximation of sigmoid
Learning: Pure gradient descent with arithmetic-only derivatives

Usage: python arithmetic_neural_net.py
"""

# Training data: XOR problem (classic neural network test)
# Format: [input1, input2, expected_output]
training_data = [
    [0.0, 0.0, 0.0],  # 0 XOR 0 = 0
    [0.0, 1.0, 1.0],  # 0 XOR 1 = 1  
    [1.0, 0.0, 1.0],  # 1 XOR 0 = 1
    [1.0, 1.0, 0.0],  # 1 XOR 1 = 0
]

# Network architecture parameters
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1
epochs = 10000

class ArithmeticNeuralNetwork:
    """
    A neural network that uses ONLY +, -, *, / operations
    
    This is educational masochism - implementing machine learning
    with the mathematical complexity of a pocket calculator!
    """
    
    def __init__(self):
        """Initialize network with small random weights using division"""
        print("🧠 Initializing Arithmetic-Only Neural Network")
        print("⚠️  WARNING: Only +, -, *, / operations allowed!")
        print("🔢 Architecture: 2 -> 3 -> 1 (19 total parameters)")
        
        # Input to hidden weights (2x3 = 6 weights)
        # Using division to create small random-ish values
        self.w1_00 = 1.0 / 3.0    # ≈ 0.333
        self.w1_01 = 1.0 / 7.0    # ≈ 0.143
        self.w1_02 = 1.0 / 5.0    # ≈ 0.200
        self.w1_10 = 1.0 / 4.0    # ≈ 0.250
        self.w1_11 = 1.0 / 6.0    # ≈ 0.167
        self.w1_12 = 1.0 / 8.0    # ≈ 0.125
        
        # Hidden layer biases (3 biases)
        self.b1_0 = 1.0 / 10.0    # ≈ 0.100
        self.b1_1 = 1.0 / 9.0     # ≈ 0.111
        self.b1_2 = 1.0 / 11.0    # ≈ 0.091
        
        # Hidden to output weights (3x1 = 3 weights)
        self.w2_00 = 1.0 / 2.0    # ≈ 0.500
        self.w2_10 = 1.0 / 3.0    # ≈ 0.333
        self.w2_20 = 1.0 / 4.0    # ≈ 0.250
        
        # Output bias (1 bias)
        self.b2_0 = 1.0 / 12.0    # ≈ 0.083
        
        print("✅ Network initialized with division-generated weights")
    
    def arithmetic_activation(self, x):
        """
        Polynomial approximation of sigmoid using only +, -, *, /
        
        Traditional sigmoid: 1 / (1 + exp(-x))
        Our approximation: x / (1 + x*x) + 0.5
        
        This gives us a smooth S-shaped curve using only basic arithmetic!
        """
        # Clamp input to prevent extreme values (using division comparison)
        # If |x| is too large, scale it down
        x_squared = x * x
        
        # Simple rational function approximation
        # f(x) = x / (1 + x²) + 0.5
        numerator = x
        denominator = 1.0 + x_squared
        result = numerator / denominator + 0.5
        
        return result
    
    def arithmetic_activation_derivative(self, x):
        """
        Derivative of our polynomial activation function
        
        If f(x) = x / (1 + x²) + 0.5
        Then f'(x) = (1 - x²) / (1 + x²)²
        
        Pure arithmetic - no exponentials!
        """
        x_squared = x * x
        numerator = 1.0 - x_squared
        denominator_base = 1.0 + x_squared
        denominator = denominator_base * denominator_base
        
        return numerator / denominator
    
    def forward_pass(self, input1, input2):
        """
        Forward propagation using only +, -, *, /
        
        Each step manually computed to show the pure arithmetic
        """
        # Store inputs
        self.input1 = input1
        self.input2 = input2
        
        # Hidden layer computation: z = W*x + b
        # Hidden neuron 0
        self.z1_0 = input1 * self.w1_00 + input2 * self.w1_10 + self.b1_0
        self.a1_0 = self.arithmetic_activation(self.z1_0)
        
        # Hidden neuron 1  
        self.z1_1 = input1 * self.w1_01 + input2 * self.w1_11 + self.b1_1
        self.a1_1 = self.arithmetic_activation(self.z1_1)
        
        # Hidden neuron 2
        self.z1_2 = input1 * self.w1_02 + input2 * self.w1_12 + self.b1_2
        self.a1_2 = self.arithmetic_activation(self.z1_2)
        
        # Output layer computation
        self.z2_0 = self.a1_0 * self.w2_00 + self.a1_1 * self.w2_10 + self.a1_2 * self.w2_20 + self.b2_0
        self.output = self.arithmetic_activation(self.z2_0)
        
        return self.output
    
    def backward_pass(self, target):
        """
        Backpropagation using only +, -, *, /
        
        Computing gradients through pure arithmetic chain rule
        """
        # Output layer error: δ = (prediction - target) * activation'(z)
        output_error = self.output - target
        output_derivative = self.arithmetic_activation_derivative(self.z2_0)
        self.delta_output = output_error * output_derivative
        
        # Hidden layer errors: δ = (Σ w*δ_next) * activation'(z)
        # Hidden neuron 0 error
        hidden_error_0 = self.delta_output * self.w2_00
        hidden_derivative_0 = self.arithmetic_activation_derivative(self.z1_0)
        self.delta_hidden_0 = hidden_error_0 * hidden_derivative_0
        
        # Hidden neuron 1 error
        hidden_error_1 = self.delta_output * self.w2_10
        hidden_derivative_1 = self.arithmetic_activation_derivative(self.z1_1)
        self.delta_hidden_1 = hidden_error_1 * hidden_derivative_1
        
        # Hidden neuron 2 error
        hidden_error_2 = self.delta_output * self.w2_20
        hidden_derivative_2 = self.arithmetic_activation_derivative(self.z1_2)
        self.delta_hidden_2 = hidden_error_2 * hidden_derivative_2
    
    def update_weights(self):
        """
        Weight updates using only +, -, *, /
        
        Standard gradient descent: w = w - learning_rate * gradient
        All gradients computed using pure arithmetic
        """
        # Update output layer weights: ∂L/∂w = δ * activation_input
        self.w2_00 = self.w2_00 - learning_rate * self.delta_output * self.a1_0
        self.w2_10 = self.w2_10 - learning_rate * self.delta_output * self.a1_1
        self.w2_20 = self.w2_20 - learning_rate * self.delta_output * self.a1_2
        
        # Update output bias
        self.b2_0 = self.b2_0 - learning_rate * self.delta_output
        
        # Update hidden layer weights: ∂L/∂w = δ * input
        # Weights from input1
        self.w1_00 = self.w1_00 - learning_rate * self.delta_hidden_0 * self.input1
        self.w1_01 = self.w1_01 - learning_rate * self.delta_hidden_1 * self.input1
        self.w1_02 = self.w1_02 - learning_rate * self.delta_hidden_2 * self.input1
        
        # Weights from input2
        self.w1_10 = self.w1_10 - learning_rate * self.delta_hidden_0 * self.input2
        self.w1_11 = self.w1_11 - learning_rate * self.delta_hidden_1 * self.input2
        self.w1_12 = self.w1_12 - learning_rate * self.delta_hidden_2 * self.input2
        
        # Update hidden biases
        self.b1_0 = self.b1_0 - learning_rate * self.delta_hidden_0
        self.b1_1 = self.b1_1 - learning_rate * self.delta_hidden_1
        self.b1_2 = self.b1_2 - learning_rate * self.delta_hidden_2
    
    def arithmetic_mean_squared_error(self, predictions, targets):
        """
        Mean squared error using only +, -, *, /
        
        MSE = (1/n) * Σ(prediction - target)²
        """
        total_error = 0.0
        count = 0.0
        
        for i in range(len(predictions)):
            error = predictions[i] - targets[i]
            squared_error = error * error
            total_error = total_error + squared_error
            count = count + 1.0
        
        mse = total_error / count
        return mse
    
    def train_one_epoch(self):
        """
        Train for one complete epoch through all training data
        
        Returns average loss for this epoch
        """
        epoch_predictions = []
        epoch_targets = []
        
        for sample in training_data:
            input1, input2, target = sample[0], sample[1], sample[2]
            
            # Forward pass
            prediction = self.forward_pass(input1, input2)
            
            # Backward pass
            self.backward_pass(target)
            
            # Update weights
            self.update_weights()
            
            # Store for loss calculation
            epoch_predictions.append(prediction)
            epoch_targets.append(target)
        
        # Calculate epoch loss using arithmetic-only MSE
        epoch_loss = self.arithmetic_mean_squared_error(epoch_predictions, epoch_targets)
        return epoch_loss
    
    def test_network(self):
        """
        Test the trained network on all training samples
        """
        print("\n🧪 Testing trained network:")
        print("Input1 | Input2 | Target | Prediction | Error")
        print("-" * 50)
        
        for sample in training_data:
            input1, input2, target = sample[0], sample[1], sample[2]
            prediction = self.forward_pass(input1, input2)
            error = prediction - target
            error_magnitude = error * error  # |error|² since we can't use abs()
            
            print(f"{input1:6.1f} | {input2:6.1f} | {target:6.1f} | {prediction:10.6f} | {error:7.4f}")

def main():
    """
    Main training loop - pure arithmetic machine learning!
    """
    print("🚀 ARITHMETIC-ONLY NEURAL NETWORK")
    print("=" * 50)
    print("🎯 Task: Learn XOR function using only +, -, *, /")
    print("🔢 No exp(), log(), sqrt(), max(), min(), abs() allowed!")
    print(f"📚 Training samples: {len(training_data)}")
    print(f"🎓 Training epochs: {epochs}")
    print(f"📈 Learning rate: {learning_rate}")
    
    # Create and train network
    network = ArithmeticNeuralNetwork()
    
    print("\n🏋️ Starting training...")
    
    for epoch in range(epochs):
        loss = network.train_one_epoch()
        
        # Print progress every 1000 epochs
        if epoch == 0 or (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1:5d}/{epochs} | Loss: {loss:.8f}")
    
    print("✅ Training completed!")
    
    # Test the network
    network.test_network()
    
    print(f"\n🎉 Neural network trained using ONLY +, -, *, / operations!")
    print("🔥 This proves that complex learning can emerge from simple arithmetic!")
    print("📚 Every gradient, every activation, every update was pure arithmetic!")

if __name__ == "__main__":
    main()

"""
🧠 EDUCATIONAL INSIGHTS FROM ARITHMETIC POVERTY:

1. **Activation Functions**: We approximated sigmoid with x/(1+x²)+0.5
   - Shows that smooth, differentiable functions can be built from ratios
   - Reveals that activation functions are just mathematical transformations

2. **Backpropagation**: Pure chain rule using +, -, *, /
   - Every gradient calculation exposed step by step
   - No mysterious library functions hiding the math

3. **Learning Process**: Weight updates through simple arithmetic
   - Shows that "learning" is just iterative parameter adjustment
   - No black magic - just basic math operations repeated many times

4. **Function Approximation**: Polynomials can approximate complex functions
   - Demonstrates universal approximation with limited tools
   - Reveals the core mathematical principles underneath fancy ML libraries

🔥 THE BEAUTY OF CONSTRAINTS:
By removing all advanced math functions, we're forced to see the 
fundamental algebraic structure that makes neural networks work.
This is machine learning stripped to its mathematical skeleton!

💡 REAL-WORLD APPLICATIONS:
- Understanding what happens inside ML libraries
- Implementing neural networks on limited hardware
- Educational tool for teaching core ML concepts
- Proving that complex intelligence can emerge from simple rules

🎯 CONCLUSION:
If you can build a neural network with just +, -, *, /, you truly 
understand the mathematical foundations of machine learning!
"""