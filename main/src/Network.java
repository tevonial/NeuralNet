import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;

/**
 * Created by Connor on 6/26/2016.
 */
public class Network {
    private static List<Layer> layers = new ArrayList<>();
    private static Layer inputLayer;
    public static double LEARNING_RATE = 1;

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
        inputLayer = new Layer(layers.size(), inputSize, 1);
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
        inputLayer.feedForward(inputList);

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

    public static void printWeights() {
        String f = "%8.4f";

        System.out.print("\n\nLayer 0: ");
        for (Neuron n : inputLayer.getNeurons()) {
           for (Double d : n.getWeights()) {
               System.out.format(f, d);
           }
        }
        System.out.println(); int i = 0;
        for (Layer l : layers) {
            System.out.print("Layer " + ++i + ": ");
            for (Neuron n : l.getNeurons()) {
                for (Double d : n.getWeights()) {
                    System.out.format(f, d);
                }
            }
            System.out.println();
        }
    }
}
