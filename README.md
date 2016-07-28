A simple set of Java classes that, together, can create a simple feedforward neural network with the ability to learn using gradient descent.

#### Construction:
```java
Network.buildFullyConnectedNetwork(INPUT, OUTPUT, HIDDEN_LAYERS, HIDDEN_WIDTH);

Network.buildConvolutionalNetwork(FEATURES, FEATURE_WIDTH, OUTPUT);
```

```java
setLearningRate(LEARNING_RATE);
```


#### Example Usage:
```java
Network net = Network.buildConvolutionalNetwork(22, 5, 10);
net.setLearningRate(0.01);

output = net.process(input);
```
or to enable backpropagation,
```java
output = net.process(input, target, true, null);
```