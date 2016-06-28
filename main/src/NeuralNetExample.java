import tevonial.neural.Network;

/**
 * Created by Connor on 6/26/2016.
 */
public class NeuralNetExample {

    private static double[][] TARGET = {
            {0.2738, 0.2234, 0.32},
            {0.1738, 0.1234, 0.22},
            { .55,    .88,    .05},
            {0.123456789},
            {0.987654321}
    };

    private static double[][] INPUT = {
            {10.83456, 0.3461, 60.342, -0.4779, 6.583},
            {1.6,     9.0,    0.29,   7.81,  -0.15},
            {0.7,     -10.2,    0.005,  -93.97,   0.09},
            {0.1, 10},
            {-99, -23}
    };

    public static void main(String[] args) {
        Network net = new Network(5, 3)
                .setHiddenLayers(1, 15)
                .build();

        for (int i = 0; i<250000; i++) {
            //net.process(INPUT[3], TARGET[3]);
            //net.process(INPUT[4], TARGET[4]);
            net.process(INPUT[0], TARGET[0]);
            net.process(INPUT[1], TARGET[1]);
            net.process(INPUT[2], TARGET[2]);
        }
        net.printWeights();
    }
}