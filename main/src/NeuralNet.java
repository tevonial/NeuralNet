/**
 * Created by Connor on 6/26/2016.
 */
public class NeuralNet {

    private static double[][] TARGET = {{0.1738, 0.1234, 0.22},{.55,.88,.05}};

    private static double[][] INPUT = {{0.83456, 0.3461, 0.342, -0.4779, 0.583},
                                       {0.6, -0.15, 0.29, 0.81, -0.15}};

    public static void main(String[] args) {
        Network net = new Network(5, 3)
                .setHiddenLayers(3, 8)
                .build();

        for (int i = 0; i<100; i++) {
            //net.process(INPUT[0], TARGET[0]);
            net.process(INPUT[1], TARGET[1]);
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
