import java.util.ArrayList;
import java.util.List;

/**
 * Created by Connor on 6/26/2016.
 */
public class Neuron {
    private int index;
    private List<Double> weights = new ArrayList<>();
    private double o;
    private double recursiveDelta;

    public Neuron(int index, int numInputs) {
        this.index = index;
        for (int i=0; i<numInputs; i++) {
            weights.add(Math.random() - Math.random());
        }
    }

    public double getOutput(List<Double> inputs) {
        double activation = 0;

        for (int i=0; i < weights.size(); i++) {
            activation += weights.get(i) * inputs.get(i);
        }

        o = (1/( 1 + Math.exp(-1*activation)));
        return o;
    }

    public void correct(int layerIndex, List<Double> d, List<Double> w) {
        if (layerIndex == 0) {  //OUTPUT LAYER
            recursiveDelta = (o - Network.TARGET[index]) * (o) * (1 - o);
        } else {                //HIDDEN LAYER
            recursiveDelta = 0;
            for (int i=0; i< d.size(); i++) {
                recursiveDelta += (d.get(i) * w.get(i));
            }
            recursiveDelta *= (o) * (1 - o);
        }

        //FINAL DELTA
        double delta = recursiveDelta * (-1) * Network.LEARNING_RATE;

        for (int i=0; i<weights.size(); i++) {
            weights.set(i, weights.get(i) + delta);
        }
    }

    public List<Double> getWeights() {
        return weights;
    }

    public double getRecursiveDelta() {
        return recursiveDelta;
    }
}
