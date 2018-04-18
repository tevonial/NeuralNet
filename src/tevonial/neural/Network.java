package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

public class Network {
    List<Layer> layers;
    double[] target;
    double learningRate;
    public int dimX, dimY;

    public Network() {
        learningRate = 0.1;
        dimX = dimY = 28;
        layers = new ArrayList<>();
    }

    public Network setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public Network setDim(int dimX, int dimY) {
        this.dimX = dimX;
        this.dimY = dimY;
        return this;
    }

    public static Network buildFullyConnectedNetwork(int inputSize, int outputSize, int hiddenLayers, int hiddenSize) {
        Network net = new Network();

        net.layers.add(new FullLayer(net, 0, outputSize, hiddenSize));
        for (int i = 0; i < hiddenLayers; i++)
            net.layers.add(new FullLayer(net, i + 1, hiddenSize, (i + 1 == hiddenLayers) ? inputSize : hiddenSize));

        net.layers.add(new FullLayer(net, net.layers.size(), inputSize, 1));

        return net;
    }

    public static Network buildConvolutionalNetwork(int numFeatures, int featureDimX, int featureDimY, int outputSize) {
        Network net = new Network();

        int convOutputSize = ((net.dimX - featureDimX + 1) / 2) * ((net.dimY - featureDimY + 1) / 2) * numFeatures;
        net.layers.add(new FullLayer(net, net.layers.size(), outputSize, convOutputSize));
        net.layers.add(new ConvolutionalLayer(net, net.layers.size(), numFeatures, featureDimX, featureDimY));
        net.layers.add(new FullLayer(net, net.layers.size(), (int) Math.pow(net.dimX, 2), 1));

        return net;
    }

    public double[] process(double[] input) {
        return process(input, null, false, null);
    }

    public double[] process(double[] input, double[] target, boolean backprop, Integer set) {
        List<Double> _input = new ArrayList<>();
        this.target = target;
        double error = 0;

        for (double i : input)
            _input.add(i);

        long start = System.nanoTime();
        layers.get(layers.size() - 1).feedForward(_input, backprop);
        long end = System.nanoTime();

        List<Double> o = ((FullLayer) layers.get(0)).output;
        double[] output = new double[o.size()];

        for (int i = 0; i < o.size(); i++) {
            output[i] = o.get(i);
            error += Math.abs(target[i] - output[i]);
        }

        printResults(output, error, (end - start) / 1000000, set);

        return output;
    }

    private void printResults(double[] output, double error, double time, Integer set) {
        if (set != null) {
            if (set < 10) System.out.print(" ");
            System.out.print(set + " -->\t");

            String f = "%3.4f";
            for (double o : output) {
                o = Math.round(o * 100000) / 10000.0;
                if (o < 100) System.out.print(' ');
                if (o < 10) System.out.print(' ');
                if (o >= 0) System.out.print(' ');
                System.out.format(f + ' ', o);
            }

            if (error < 0)
                System.out.format("\terror: " + f, error);
            else
                System.out.format("\terror:  " + f, error);

            System.out.println(" \t" + time + " ms");
        }
    }

    static double activate(double x) {
        return (1 / ( 1 + Math.exp(-1 * x)));                       //SIGMOID
        //return Math.log(1 + Math.exp(x));                     //ReLU

        /*if (x <= 0) {                                         //Leaky ReLU
            return x * 0.01;
        } else {
            return x;
        }*/
    }

    static double activatePrime(double x) {
        return x * (1.0 - x);                                  //SIGMOID
        //return  1.0 / (1.0 + Math.exp(-1 * x));              //ReLU

        /*if (x <= 0) {                                        //Leaky ReLU
            return  0.01;
        } else {
            return 1;
        }*/
    }
}
