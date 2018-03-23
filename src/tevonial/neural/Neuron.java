package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    private Network network;
    private int index;
    private int numInputs;
    private List<Double> weights;
    private List<Double> input;
    private double o;
    private double d;

    public Neuron() {}

    public Neuron(Network network, int index, int numInputs) {
        this.network = network;
        this.index = index;
        this.numInputs = numInputs;
        weights = new ArrayList<>();
        for (int i=0; i<=numInputs; i++) {  //<= for bias
            weights.add((Math.random() - Math.random()) / 1.0);
        }
    }

    public double getOutput(List<Double> inputs) {
        double activation = 0;

        if (numInputs == 1) {                               //For input layer
            this.input = new ArrayList<>();
            this.input.add(inputs.get(index));
        } else {                                            //For all other layers
            this.input = inputs;
        }

        for (int i = 0; i < input.size(); i++) {
            try {
                activation += weights.get(i) * input.get(i);
            } catch (IndexOutOfBoundsException e) {
                System.err.println("weights.size()=" + weights.size() + "\tinput.size()=" + input.size());
            }
        }

        activation += weights.get(weights.size()-1);        //bias weight * 1

        o = Network.activate(activation);

        return o;
    }

    public void correct(double E) {
        this.d = E * Network.activatePrime(o);

        //FINAL DELTA
        double delta = this.d * (-1) * network.LEARNING_RATE;

        for (int i=0; i<weights.size(); i++) {
            double deltaWeight = delta;
            try {
                deltaWeight *= input.get(i);
            } catch (IndexOutOfBoundsException e) {}

            weights.set(i, weights.get(i) + deltaWeight);
        }
    }



    public List<Double> getWeights() {
        return weights;
    }

    public double getDelta() {
        return d;
    }
}
