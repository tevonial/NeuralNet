import java.util.ArrayList;
import java.util.List;

/**
 * Created by Connor on 6/26/2016.
 */
public class Layer {
    private List<Neuron> neurons = new ArrayList<>();
    private List<Double> output = new ArrayList<>();
    private int index;

    public Layer(int index, int size, int numInputs) {
        this.index = index;
        for (int i=0; i<size; i++) {
            neurons.add(new Neuron(index, numInputs));
        }
    }

    public void feed(List<Double> input) {
        output.clear();
        for (int i=0; i<neurons.size(); i++) {
            output.add(neurons.get(i).getOutput(input));
        }

        Layer next = Network.getLayer(index - 1);
        if (next == null) {
            double[] errors = new double[neurons.size()];
            for (int i=0; i<Network.target.length; i++) {
                errors[i] = Network.target[i] - output.get(i);
            }
            correct(errors);
        } else {
            next.feed(output);
        }
    }

    public void correct(double[] errors) {
        int i = 0;
        for (Neuron n : neurons) {
            n.correct(errors[i++], index);
        }
    }

    public List<Double> getOutput() {
        return output;
    }
}
