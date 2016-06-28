import tevonial.neural.Network;

/**
 * Created by Connor on 6/26/2016.
 */
public class NeuralNetExample {

    private static double[][] TARGET = {
            {0.2738, 0.2234, 0.32},
            {0.1738, 0.1234, 0.22},
            { .55,    .88,    .05},
            {0.4326, 0.857,  0.98}
    };

    private static double[][] INPUT = {
            {0.83456, 0.3461, 0.342, -0.4779, 0.583},
            {0.6,     0.0,    0.29,   0.81,  -0.15},
            {0.7,     0.2,    0.005,  0.97,   0.09}
    };

    public static void main(String[] args) {
        Network net = new Network(2, 3)
                .setHiddenLayers(8, 15)
                .build();

        for (int i = 0; i<1000; i++) {
            net.process(INPUT[0], TARGET[0]);
            net.process(INPUT[1], TARGET[1]);
            net.process(INPUT[2], TARGET[2]);
            //try {
                //Thread.sleep(10);
            //} catch (InterruptedException e) {
            //    e.printStackTrace();
            //}
        }
        net.printWeights();
    }
}