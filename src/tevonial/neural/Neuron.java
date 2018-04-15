package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

class Neuron {
    private Network network;
    private int inputs, index;
    List<Double> input, weights;
    private double output;
    double delta;

    Neuron(Network network, int index, int inputs) {
        this.network = network;
        this.inputs = inputs;
        this.index = index;
        input = new ArrayList<>();
        weights = new ArrayList<>();

        for (int i = 0; i <= inputs; i++)  //<= for bias
            weights.add(Math.random() - Math.random());
    }

    double filter(List<Double> in) {
        double activation = weights.get(weights.size() - 1);    // bias weight * 1;

        if (inputs == 1)                                        // For input layer
            input.add(in.get(index));
        else                                                    // For all other layers
            input = in;

        for (int i = 0; i < Math.min(input.size(), weights.size()); i++)
            activation += weights.get(i) * input.get(i);

        output = Network.activate(activation);

        return output;
    }

    void correct(double E) {
        this.delta = E * Network.activatePrime(output);
        double d = this.delta * network.learningRate * -1;

        for (int i = 0; i < Math.min(input.size(), weights.size()); i++)
            weights.set(i, weights.get(i) + d * input.get(i));
    }
}
