package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

public class Network {

    List<Layer> layers;
    private double[] target;

    double LEARNING_RATE;
    public int WIDTH = 28, HEIGHT = 28;

    public Network() {}

    public void setLearningRate(double l) {
        LEARNING_RATE = l;
    }

    public static Network buildFullyConnectedNetwork(int inputSize, int outputSize, int hiddenLayers, int hiddenSize) {
        Network net = new Network();

        net.layers = new ArrayList<>();

        net.layers.add(new FullLayer(net, 0, outputSize, hiddenSize));
        for (int i = 0; i < hiddenLayers; i++) {
            net.layers.add(new FullLayer(net, i + 1, hiddenSize, (i + 1 == hiddenLayers) ? inputSize : hiddenSize));
        }
        net.layers.add(new FullLayer(net, net.layers.size(), inputSize, 1));

        return net;
    }

    public static Network buildConvolutionalNetwork(int numFeatures, int featureDim, int outputSize) {
        Network net = new Network();

        net.layers = new ArrayList<>();

        int convOutputSize = ((int) Math.pow(((net.WIDTH - featureDim + 1) / 2), 2)) * numFeatures;

        net.layers.add(new FullLayer(net, net.layers.size(), outputSize, convOutputSize));
        net.layers.add(new ConvolutionalLayer(net, net.layers.size(), numFeatures, featureDim));

        return net;
    }

    public Layer getLayer(int index) {
        if (index == -1 || index >= layers.size()) {
            return null;
        } else {
            return layers.get(index);
        }
    }

    public double getTarget(int index) {
        return target[index];
    }

    public double[] process(double[] input) {
        return process(input, null, false, null);
    }

    public double[] process(double[] input, double[] target, boolean backprop, Integer digit) {
        List<Double> inputList = new ArrayList<>();
        for (int i=0; i<input.length; i++) {
            inputList.add(input[i]);
        }

        this.target = target;
        layers.get(layers.size()-1).feedForward(inputList, backprop);

        List<Double> o = ((FullLayer) layers.get(0)).getOutput();
        double[] output = new double[o.size()];
        for (int i=0; i<o.size(); i++) {
            output[i] = o.get(i);
        }

        if (digit != null) {
            printResults(output, "O" + digit);
        }

        return output;
    }

    public void printResults(double[] output, String set) {
        String f = "%3.2f  ";
        System.out.print(set + " --> ");
        for (double out : output) {
            System.out.format(f, out*100.0);
        }
        System.out.println();
    }

    public static double activate(double x) {
        return (1/( 1 + Math.exp(-1*x)));                       //SIGMOID
        //return Math.log(1 + Math.exp(x));                     //ReLU

        /*if (x <= 0) {                                         //Leaky ReLU
            return x * 0.01;
        } else {
            return x;
        }*/
    }

    public static double activatePrime(double x) {
        return x * (1.0 - x);                                  //SIGMOID
        //return  1.0 / (1.0 + Math.exp(-1 * x));              //ReLU

        /*if (x <= 0) {                                        //Leaky ReLU
            return  0.01;
        } else {
            return 1;
        }*/
    }
}
