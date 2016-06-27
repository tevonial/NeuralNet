import java.util.*;

/**
 * Created by Connor on 6/26/2016.
 */
public class Layer {
    private List<Neuron> neurons = new ArrayList<>();
    private List<Double> output = new ArrayList<>();
    private int layerIndex;

    public Layer(int layerIndex, int size, int numInputs) {
        this.layerIndex = layerIndex;
        for (int i=0; i<size; i++) {
            neurons.add(new Neuron(i, numInputs));
        }
    }

    public void feedForward(List<Double> input) {
        output.clear();
        for (int i=0; i<neurons.size(); i++) {
            output.add(neurons.get(i).getOutput(input));
        }

        Layer next = Network.getLayer(layerIndex - 1);
        if (next == null) {
            this.backPropagate(null, null);
        } else {
            next.feedForward(output);
        }
    }

    public void backPropagate(List<Double> d, List<List<Double>> w) {
        List<List<Double>> _w = new ArrayList<>();
        List<Double> _d = new ArrayList<>();
        for (int j=0; j<neurons.size(); j++) {
            Neuron n = neurons.get(j);

            try {
                n.correct(layerIndex, d, w.get(j));
            } catch (NullPointerException e){
                n.correct(layerIndex, null, null);
            }

            _d.add(n.getRecursiveDelta());
            _w.add(n.getWeights());
        }

        _w = rotate(_w);

        try {
            Network.getLayer(layerIndex + 1).backPropagate(_d, _w);
        } catch (NullPointerException e) {}
    }

    public List<Double> getOutput() {
        return output;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    private List<List<Double>> rotate(List<List<Double>> input) {
        List<List<Double>> w = new ArrayList<>(); boolean first = true;
        List<Double> sample = new ArrayList<>();
        for (int p=0; p<input.size(); p++) {
            sample.add(0.0);
        }

        for (int i=0; i<input.size(); i++) {
            List<Double> l = input.get(i);
            if (first) {
                for (int x = 0; x < l.size(); x++) {
                    w.add(sample);
                }
            }
            first = false;

            for (int x = 0; x < l.size(); x++) {
                w.get(x).set(i, l.get(x));
            }
        }
        return w;
    }
}

