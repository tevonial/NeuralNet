package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

public class FeatureMap {
    private int dim;
    private ConvolutionalLayer layer;

    private List<Double> weights;
    private List<Double> out;

    private int convolutedDim;

    List<Double> _E = new ArrayList<>();

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
        out = new ArrayList<>();

        for (int y = 0; y < convolutedDim; y++) {
            for (int x = 0; x < convolutedDim; x++) {

                double activation = 0.0;

                for (int i = 0; i < dim*dim; i++) {
                    activation += weights.get(i) * input.get( (((i/dim) + y) * 28) + (i%dim + x) );
                }

                activation += weights.get(weights.size()-1);        //bias weight * 1

                out.add(Network.activate(activation));
            }
        }

        return out;
    }

    public void correct(List<Double> E) {

        for (int a=0; a<dim; a++) {
            for (int b = 0; b<= dim; b++) {

                double dweight = 0.0;

                for (int i=0; i<convolutedDim; i++) {
                    for (int j=0; j<convolutedDim; j++) {

                        double e = E.get(((i/2)*(convolutedDim/2)) + j/2);

                        try {
                            dweight += e * Network.activatePrime(out.get((i*convolutedDim) + j)) * layer.input.get(((i + a) * 28) + (j + b));
                        } catch (Exception ex) {
                            dweight += e * Network.activatePrime(out.get((i*convolutedDim) + j));
                        }

                    }
                }

                dweight *= (-1) * layer.network.LEARNING_RATE;

                weights.set(a*dim + b, weights.get(a*dim + b) + dweight);

            }
        }
    }
}