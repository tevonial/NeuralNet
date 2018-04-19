package tevonial.test;

import tevonial.neural.Network;

public class NeuralNetTest {
    private static double[][] target = {
            {0.2738, 0.2234, 0.3218, 0.8765},
            {0.1738, 0.1234, 0.2209, 0.1922},
            {0.5950, 0.8807, 0.1599, 0.6775},
            {0.2771, 0.6402, 0.6156, 0.0609}
    };

    public static void main(String[] args) {
        Network nets[] = {
            Network.buildFullyConnectedNetwork(20, 3, 25, 4),
            Network.buildConvolutionalNetwork(5,    5   ,  5, 4)
        };

        double input[][] = new double[nets.length][];
        input[0] = new double[20];
        input[1] = new double[nets[1].dimX * nets[1].dimY];
        int iterations = 200;

        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < nets.length; j++) {
                for (int k = 0; k < input[j].length; k++)
                    input[j][k] = Math.random();
            }

            for (int j = 0; j <= iterations; j++) {
                for (int k = 0; k < nets.length; k++)
                    nets[k].process(input[k], target[k], true, (j == iterations) ? k : null);
            }
        }
    }
}