import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

/**
 * Created by Connor on 6/26/2016.
 */
public class Neuron {
    private List<Double> weights = new ArrayList<>();
    private List<Double> last = new ArrayList<>();
    private int index;

    public Neuron(int index, int numInputs) {
        this.index = index;
        for (int i=0; i<numInputs; i++) {
            weights.add(Math.random() - Math.random());
        }
    }

    public double getOutput(List<Double> inputs) {
        last = inputs;
        double activation = 0;

        for (int i=0; i < weights.size(); i++) {
            activation += weights.get(i) * inputs.get(i);
        }

        return (1/( 1 + Math.exp(-1*activation)));
    }

    public void correct(double error, int degree) {
        double[] errors = new double[weights.size()];

        for (int i=0; i<weights.size(); i++) {
            errors[i] = error / last.get(i);
            weights.set(i, weights.get(i) + (errors[i] / (1*Math.exp(degree))));
        }

        try {
            Network.getLayer(index + 1).correct(errors);
        } catch (NullPointerException e) {

        }
    }
}
