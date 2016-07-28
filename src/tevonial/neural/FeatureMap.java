package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

public class FeatureMap {
    private int dim;
    private ConvolutionalLayer layer;

    private List<Double> weights;
    private List<Double> out;

    private int convolutedDim;

    public FeatureMap() {}

    public FeatureMap(ConvolutionalLayer layer, int dim) {
        this.dim = dim;
        this.layer = layer;
        convolutedDim = layer.convolutedDim;

        weights = new ArrayList<>();
        for (int i=0; i <= dim*dim; i++) {  //<= for bias
            weights.add( (Math.random() - Math.random()) / 5.0);
        }
    }

    public List<Double> filter(List<Double> input) {
        List<Double> in;
        out = new ArrayList<>();

        for (int y = 0; y < convolutedDim; y++) {
            for (int x = 0; x < convolutedDim; x++) {

                in = new ArrayList<>();
                for (int y2 = 0; y2 < dim; y2++) {
                    for (int x2 = 0; x2 < dim; x2++) {
                        in.add(input.get(((y + y2) * 28) + (x + x2)));
                    }
                }

                double activation = 0.0;

                for (int i = 0; i < in.size(); i++) {
                    activation += weights.get(i) * in.get(i);
                }

                activation += weights.get(weights.size()-1);        //bias weight * 1

                out.add(Network.activate(activation));
            }
        }

        return out;
    }

    public void backPropagate(List<Double> d, List<List<Double>> w) {

        for (int a=0; a<dim; a++) {
            for (int b = 0; b<= dim; b++) {

                double delta = 0.0;

                for (int i=0; i<convolutedDim; i++) {
                    for (int j=0; j<convolutedDim; j++) {

                        List<Double> _w = w.get(((i/2)*(convolutedDim/2)) + j/2);       //depool

                        for (int q=0; q<d.size(); q++) {
                            try {
                                delta += d.get(q) * _w.get(q) * layer.input.get(((i + a) * 28) + (j + b));
                            } catch (Exception e) {
                                delta += d.get(q) * _w.get(q);
                            }
                        }

                        double x = out.get((i*convolutedDim) + j);

                        delta *= Network.activatePrime(x) * (-1) * layer.network.LEARNING_RATE;

                    }
                }

                weights.set(a*dim + b, weights.get(a*dim + b) + delta);

            }
        }
    }

}