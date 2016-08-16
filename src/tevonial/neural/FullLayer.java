package tevonial.neural;

import java.util.*;

public class FullLayer implements Layer {
    private Network network;
    private List<Neuron> neurons;
    private List<Double> output;
    private int layerIndex;

    public FullLayer() {}

    public FullLayer(Network network, int layerIndex, int size, int numInputs) {
        this.network = network;
        this.layerIndex = layerIndex;
        neurons = new ArrayList<>();
        for (int i=0; i<size; i++) {
            neurons.add(new Neuron(network, i, numInputs));
        }
    }

    @Override
    public void feedForward(List<Double> input, boolean backprop) {
        output = new ArrayList<>();
        for (int i=0; i<neurons.size(); i++) {
            output.add(neurons.get(i).getOutput(input));
        }

        try {
            network.getLayer(layerIndex - 1).feedForward(output, backprop);
        } catch (NullPointerException e) {
            if (backprop) this.backPropagate(null);
        }

    }

    @Override
    public void backPropagate(List<Double> E) {
        List<List<Double>> w = new ArrayList<>();
        List<Double> d = new ArrayList<>();

        for (int j=0; j<neurons.size(); j++) {
            Neuron n = neurons.get(j);

            if (layerIndex == 0) {
                n.correct(output.get(j) - network.getTarget(j));
            } else {
                n.correct(E.get(j));
            }

            d.add(n.getDelta());
            w.add(n.getWeights());
        }

        //Calculate error for previous layer, no need to rotate
        E = new ArrayList<>();
        for (int i=0; i<w.get(0).size(); i++) {
            double e = 0.0;
            for (int j=0; j<d.size(); j++) {
                e += d.get(j) * w.get(j).get(i);
            }
            E.add(e);
        }

        try {
            network.getLayer(layerIndex + 1).backPropagate(E);
        } catch (NullPointerException e) {}
    }

    public List<Double> getOutput() {
        return output;
    }

}

