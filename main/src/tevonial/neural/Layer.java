package tevonial.neural;

import java.util.*;

/**
 * Created by Connor on 6/26/2016.
 */
public class Layer {
    private Network network;
    private List<Neuron> neurons;
    private List<Double> output;
    private int layerIndex;

    public Layer(Network network, int layerIndex, int size, int numInputs) {
        this.network = network;
        this.layerIndex = layerIndex;
        neurons = new ArrayList<>();
        for (int i=0; i<size; i++) {
            neurons.add(new Neuron(network, i, numInputs));
        }
    }

    public void feedForward(List<Double> input) {
        output = new ArrayList<>();
        for (int i=0; i<neurons.size(); i++) {
            output.add(neurons.get(i).getOutput(input));
        }

        Layer next = network.getLayer(layerIndex - 1);
        if (next == null) {
            this.backPropagate(null, null);
        } else {
            next.feedForward(output);
        }
    }

    private void backPropagate(List<Double> d, List<List<Double>> w) {
        List<List<Double>> _w = new ArrayList<>();
        List<Double> _d = new ArrayList<>();
        for (int j=0; j<neurons.size(); j++) {
            Neuron n = neurons.get(j);

            try {
                n.correct(layerIndex, d, w.get(j));
            } catch (NullPointerException e){
                n.correct(layerIndex, null, null);
            }

            _d.add(n.getD());
            _w.add(n.getWeights());
        }

        _w = rotate(_w);

        try {
            network.getLayer(layerIndex + 1).backPropagate(_d, _w);
        } catch (NullPointerException e) {}
    }

    public List<Double> getOutput() {
        return output;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    private List<List<Double>> rotate(List<List<Double>> in) {
        List<List<Double>> out = new ArrayList<>(); boolean first = true;
        List<Double> row = new ArrayList<>();
        for (int i=0; i<in.size(); i++) {
            row.add(null);
        }

        for (int x=0; x<in.size(); x++) {
            List<Double> l = in.get(x);
            if (first) {
                for (int i=0; i<l.size(); i++) {
                    out.add(row);
                }
            }
            first = false;

            for (int y=0; y<l.size(); y++) {
                out.get(y).set(x, l.get(y));
            }
        }
        return out;
    }
}

