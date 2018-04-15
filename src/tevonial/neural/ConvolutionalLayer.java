package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalLayer implements Layer {
    Network network;
    int index, factorX, factorY;
    private int convolutedDimX, convolutedDimY, featureDimX, featureDimY;
    List<Double> input, E;
    private List<FeatureMap> features;

    ConvolutionalLayer(Network network, int index, int numFeatures, int featureDimX, int featureDimY) {
        this.network = network;
        this.index = index;
        this.convolutedDimX = network.dimX - featureDimX + 1;
        this.convolutedDimY = network.dimY - featureDimY + 1;
        this.featureDimX = featureDimX;
        this.featureDimY = featureDimY;
        this.factorX = this.factorY = 2;

        features = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++)
            features.add(new FeatureMap(this, featureDimX, featureDimY));
    }

    @Override
    public void feedForward(List<Double> input, boolean backprop) {
        this.input  = input;
        List<Double> output = new ArrayList<>();

        for (FeatureMap map : features)
            output.addAll(pool(map.filter(input)));

        network.layers.get(index - 1).feedForward(output, backprop);
    }

    @Override
    public void backPropagate(List<Double> E) {
        for (int f = 0; f < features.size(); f++) {
            int convolutedSize = (convolutedDimX * convolutedDimY) / (factorX * factorY);
            int index = f * convolutedSize;

            features.get(f).correct(E.subList(index, index + featureDimY * featureDimX));
        }

//        TODO: caltulate deltas
//
//        for (int i = 0; i < w.get(0).size(); i++) {
//            double e = 0.0;
//            for (int j = 0; j < d.size(); j++)
//                e += d.get(j) * w.get(j).get(i);
//            E.add(e);
//        }
//
//        if (index != network.layers.size() - 1)
//              network.getLayer(layerIndex + 1).backPropagate(E);
    }

    private List<Double> pool(List<Double> input) {
        List<Double> out = new ArrayList<>();

        if (input.size() == 1)
            return input;

        int poolDimX = (convolutedDimX - factorX) / factorX, poolDimY = (convolutedDimX - factorX) / factorX;

        for (int y = 0; y < poolDimY; y += factorY) {
            for (int x = 0; x < poolDimX; x += factorX) {
                double max = 0.0;

                for (int y2 = 0; y2 < factorY; y2++)
                    for (int x2 = 0; x2 < factorX; y2++) {
                        max = Math.max(max, input.get((y + y2) * convolutedDimX + x + x2));
                        out.add(max);
                    }
            }
        }

        return out;
    }
}
