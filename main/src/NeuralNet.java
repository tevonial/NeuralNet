/**
 * Created by Connor on 6/26/2016.
 */
public class NeuralNet {

    private static double[][] TARGET = {{0.1738, 0.1234, 0.22},{1}};

    private static double[][] INPUT = {{0.83456, 0.3461, 0.342, 0.4779, 0.583},
                                       {0.6, 0.15, 0.29, 0.81, 0.15}};

    public static void main(String[] args) {
        Network net = new Network(5, 3)
                .setHiddenLayers(3, 8)
                .build();

        while (true) {
            net.process(INPUT[1], TARGET[0]);
        }
    }

    public static void printResults(double[] output, double[] target) {
        for (double out : output) {
            System.out.print(out + " ");
        }
        System.out.println();
    }
}
