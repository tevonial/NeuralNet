package tevonial.test;

import tevonial.neural.Network;

public class NeuralNetTest {

    private static double[][] TARGET = {
            {0.2738, 0.2234, 0.32},
            {0.1738, 0.1234, 0.22},
            {0.5950, 0.8807, 0.05},
            {.2771,   .6402,    .81}
    };

    private static double[][] INPUT = {
            {10.83456, 0.3461, 60.342, -0.4779, 6.583},
            {1.6,     9.0,    0.29,   7.81,  -0.15},
            {0.7,     -10.2,    0.005,  -93.97,   0.09}
    };

    public static void main(String[] args) {
        Network nets[] = {
            Network.buildFullyConnectedNetwork(5, 3, 3, 25),
            Network.buildFullyConnectedNetwork(5, 3, 3, 25),
            Network.buildFullyConnectedNetwork(5, 3, 3, 25),
            Network.buildConvolutionalNetwork(5,    7,  3)
        };

        double input3[] = new double[nets[3].DIM * nets[3].DIM];
        for (int i = 0; i<input3.length; i++) {
            input3[i] = Math.random();
        }

        int ITERATIONS = 200;
        for (int j = 0; j < 20; j++) {
            for (int i = 0; i <= ITERATIONS; i++) {
//            nets[0].process(INPUT[0], TARGET[0], true, (i == ITERATIONS) ? 1 : null);
//            nets[1].process(INPUT[1], TARGET[1], true, (i == ITERATIONS) ? 2 : null);
//            nets[2].process(INPUT[2], TARGET[2], true, (i == ITERATIONS) ? 3 : null);
                nets[3].process(input3, TARGET[3], true, (i == ITERATIONS) ? 4 : null);
            }
        }
    }
}