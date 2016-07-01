package tevonial.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Neuron implements Serializable {
    private Network network;
    private int index;
    private int numInputs;
    private List<Double> weights;
    private List<Double> input;
    private double o;
    private double d;

    public Neuron(Network network, int index, int numInputs) {
        this.network = network;
        this.index = index;
        this.numInputs = numInputs;
        weights = new ArrayList<>();
        for (int i=0; i<=numInputs; i++) {  //<= for bias
            weights.add(Math.random() - Math.random());
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
            activation += weights.get(i) * input.get(i);
        }

        activation += weights.get(weights.size()-1);        //bias weight * 1

        o = (1/( 1 + Math.exp(-1*activation)));
        return o;
    }

    public void correct(int layerIndex, List<Double> d, List<Double> w) {
        if (layerIndex == 0) {  //OUTPUT LAYER
            this.d = (o - network.getTarget(index)) * (o) * (1 - o);
        } else {                //HIDDEN LAYER
            this.d = 0;
            for (int i=0; i< d.size(); i++) {
                this.d += (d.get(i) * w.get(i));
            }
            this.d *= (o) * (1 - o);
        }

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

    public double getD() {
        return d;
    }
}
