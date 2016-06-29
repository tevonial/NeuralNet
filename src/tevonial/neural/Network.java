package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Connor on 6/26/2016.
 */
public class Network {
    private List<Layer> layers;
    private Layer inputLayer;
    private int hiddenSize, inputSize, outputSize, hiddenLayers;
    private double[] target;

    public double LEARNING_RATE = .8;

    public Network(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public Network setHiddenLayers(int layers, int hiddenSize) {
        this.hiddenLayers = layers;
        this.hiddenSize = hiddenSize;
        return this;
    }

    public Network build() {
        layers = new ArrayList<>();
        layers.add(new Layer(this, 0, outputSize, hiddenSize));
        for (int i=0; i<hiddenLayers; i++) {
            layers.add(new Layer(this, i+1, hiddenSize, (i+1 ==  hiddenLayers) ? inputSize : hiddenSize));
        }
        inputLayer = new Layer(this, layers.size(), inputSize, 1);
        return this;
    }

    public Layer getLayer(int index) {
        if (index == layers.size()) {
            return inputLayer;
        } else if (index == -1 || index >= layers.size()) {
            return null;
        } else {
            return layers.get(index);
        }
    }

    public double getTarget(int index) {
        return target[index];
    }

    public void process(double[] input, double[] target, String set) {
        List<Double> inputList = new ArrayList<>();
        for (int i=0; i<input.length; i++) {
            inputList.add(input[i]);
        }

        this.target = target;
        inputLayer.feedForward(inputList);

        List<Double> o = layers.get(0).getOutput();
        double[] output = new double[o.size()];
        for (int i=0; i<o.size(); i++) {
            output[i] = o.get(i);
        }

        if (set != null) {
            printResults(output, set);
        }
    }

    private static void printResults(double[] output, String set) {
        String f = "%4.10f  ";
        System.out.print(set + " --> ");
        for (double out : output) {
            System.out.format(f, out);
        }
        System.out.println();
    }

    public void printWeights() {
        String f = "%8.4f";

        System.out.print("\n\nLayer " + layers.size() + ": ");
        for (Neuron n : inputLayer.getNeurons()) {
           for (Double d : n.getWeights()) {
               System.out.format(f, d);
           }
        }
        System.out.println();
        for (int i=layers.size()-1; i>=0; i--) {
            Layer l = layers.get(i);
            System.out.print("Layer " + i + ": ");
            for (Neuron n : l.getNeurons()) {
                for (Double d : n.getWeights()) {
                    System.out.format(f, d);
                }
            }
            System.out.println();
        }
    }


}
