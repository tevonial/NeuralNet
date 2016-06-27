import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Connor on 6/26/2016.
 */
public class Network {
    private static List<Layer> layers = new ArrayList<>();
    private static Layer inputLayer;

    private int hiddenSize, inputSize, outputSize, hiddenLayers, iteration;
    public static double[] target;

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
        iteration = 0;
        layers.clear();
        layers.add(new Layer(0, outputSize, hiddenSize));
        for (int i=0; i<hiddenLayers; i++) {
            layers.add(new Layer(i+1, hiddenSize, (i+1 ==  hiddenLayers) ? inputSize : hiddenSize));
        }
        inputLayer = new Layer(layers.size(), inputSize, inputSize);
        return this;
    }

    public static Layer getLayer(int index) {
        if (index == layers.size()) {
            return inputLayer;
        } else if (index == -1 || index >= layers.size()) {
            return null;
        } else {
            return layers.get(index);
        }
    }

    public void process(double[] input, double[] target) {
        List<Double> inputList = new ArrayList<>();
        for (int i=0; i<input.length; i++) {
            inputList.add(input[i]);
        }

        this.target = target;
        inputLayer.feed(inputList);

        List<Double> output = layers.get(0).getOutput();
        double[] doutput = new double[output.size()];
        for (int i=0; i<output.size(); i++) {
            doutput[i] = output.get(i);
        }

        NeuralNet.printResults(doutput, target);


        try {
            Thread.sleep(50);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
