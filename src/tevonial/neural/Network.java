package tevonial.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Network implements Serializable {
    private List<Layer> layers;
    private Layer inputLayer;
    private int hiddenSize, inputSize, outputSize, hiddenLayers;
    private double[] target;

    public double LEARNING_RATE;

    public Network(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public Network setHiddenLayers(int layers, int hiddenSize) {
        this.hiddenLayers = layers;
        this.hiddenSize = hiddenSize;
        return this;
    }

    public void setLearningRate(double l) {
        LEARNING_RATE = l;
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

    public double[] process(double[] input, double[] target, boolean backprop, Integer digit) {
        List<Double> inputList = new ArrayList<>();
        for (int i=0; i<input.length; i++) {
            inputList.add(input[i]);
        }

        this.target = target;
        inputLayer.feedForward(inputList, backprop);

        List<Double> o = layers.get(0).getOutput();
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
        String f = "%4.10f  ";
        System.out.print(set + " --> ");
        for (double out : output) {
            System.out.format(f, out*100.0);
        }
        System.out.println();
    }
}
