/**
 * Created by Connor on 6/26/2016.
 */
public class NeuralNet {

    private static double[][] TARGET = {
            {0.1738, 0.1234, 0.22},
            { .55,    .88,    .05},
            {0.4326,0.857,0.98}
    };

    private static double[][] INPUT = {
            {0.83456, 0.3461, 0.342, -0.4779, 0.583},
            {0.6,     0.0,    0.29,   0.81,  -0.15},
            {0.7,0.2,0.005,0.97,0}
    };

    public static void main(String[] args) {
        Network net = new Network(5, 3)
                .setHiddenLayers(3, 12)
                .build();

        for (int i = 0; i<i+1; i++) {
            net.process(INPUT[0], TARGET[0]);
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        Network.printWeights();
    }

    public static void printResults(double[] output, double[] target) {
        for (double out : output) {
            System.out.print(out + " ");
        }
        System.out.println();
    }
}