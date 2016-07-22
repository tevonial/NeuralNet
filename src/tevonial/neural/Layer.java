package tevonial.neural;

import java.util.List;

public interface Layer {
    void feedForward(List<Double> input, boolean backprop);
    void backPropagate(List<Double> d, List<List<Double>> w);
}