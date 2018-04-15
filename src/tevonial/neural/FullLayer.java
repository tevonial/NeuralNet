package tevonial.neural;

import java.util.*;

public class FullLayer implements Layer {
    private Network network;
    private List<Neuron> neurons;
    List<Double> output;
    private int index;

    FullLayer(Network network, int index, int size, int inputs) {
        this.network = network;
        this.index = index;
        neurons = new ArrayList<>();

        for (int i = 0; i < size; i++)
            neurons.add(new Neuron(network, i, inputs));
    }

    @Override
    public void feedForward(List<Double> input, boolean backprop) {
        output = new ArrayList<>();
        for (Neuron neuron : neurons)
            output.add(neuron.filter(input));

        if (index > 0 || network.layers.size() == 1)
            network.layers.get(index - 1).feedForward(output, backprop);
        else
            if (backprop) this.backPropagate(null);
    }

    @Override
    public void backPropagate(List<Double> E) {
        List<List<Double>> w = new ArrayList<>();
        List<Double> d = new ArrayList<>();

        for (int i = 0; i < neurons.size(); i++) {
            Neuron n = neurons.get(i);

            if (index == 0)
                n.correct(output.get(i) - network.target[i]);
            else
                n.correct(E.get(i));

            d.add(n.delta);
            w.add(n.weights);
        }

        //Calculate error for previous layer, no need to rotate
        E = new ArrayList<>();
        for (int i = 0; i < w.get(0).size(); i++) {
            double e = 0.0;
            for (int j = 0; j < d.size(); j++)
                e += d.get(j) * w.get(j).get(i);
            E.add(e);
        }

        if (index != network.layers.size() - 1)
            network.layers.get(index + 1).backPropagate(E);
    }
}

