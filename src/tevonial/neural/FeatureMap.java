package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

class FeatureMap {
    private ConvolutionalLayer layer;

    private List<Double> weights;
    private List<Double> out;

    private int dimX, dimY;

    FeatureMap(ConvolutionalLayer layer, int featureDimX, int featureDimY) {
        this.layer = layer;
        this.dimX = featureDimX;
        this.dimY = featureDimY;

        weights = new ArrayList<>();
        for (int i = 0; i <= dimX * dimY; i++)  //<= for bias
            weights.add(Math.random());
    }

    List<Double> filter(List<Double> input) {
        out = new ArrayList<>();
        double activation = 0.0;

        for (int i = 0; i < dimX * dimY; i++)
            activation += weights.get(i) * input.get(i);

        activation += weights.get(weights.size() - 1);        //bias weight * 1
        out.add(Network.activate(activation));

        return out;
    }

    void correct(List<Double> E) {
        if (out.size() > 0) {
            for (int y = 0; y < dimY; y++) {
                for (int x = 0; x <= dimX; x++) {
                    double d = 0.0;

                    for (int i = 0; i < out.size() / dimY; i++) {
                        for (int j = 0; j < out.size() % dimX; j++) {
                            double e = E.get(((i / layer.factorY) * (dimY / layer.factorY)) + j / layer.factorX);

                            if (layer.index == layer.network.layers.size())
                                d += e * Network.activatePrime(out.get(((y + i) * dimX) + x + j)) * layer.input.get(((y + i) * dimX) + x + j);
                            else
                                d += e * Network.activatePrime(out.get(((y + i) * dimX) + x + j));
                        }
                    }

                    d *= -1 * layer.network.learningRate;
                    weights.set(y * dimX + x, weights.get(y * dimX + x) + d);
                }
            }
        }
    }
}