A simple set of Java classes that, together, can create a simple feedforward neural network with the ability to learn using gradient descent.

#### Construction:
```java
new Network(INPUT_WIDTH, OUTPUT_WIDTH)
    .setHiddenLayers(LAYERS, WIDTH)
    .build();
```

```java
setLearningRate(LEARNING_RATE);
```


#### Example Usage:
```java
Network net = new Network(3, 2)
    .setHiddenLayers(1, 2)
    .build();
net.setLearningRate(0.01);
double[] input = new double[]{1.0, 2.0, 3.0};
double[] target = new double[]{0.5, 0.5};


output = net.process(input);
```
or to enable backpropagation,
```java
output = net.process(input, target, true, null);
```