package tevonial.neural;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalLayer implements Layer {
    Network network;
    int layerIndex;
    int convolutedDim;

    List<Double> input;
    private List<FeatureMap> features;

    public ConvolutionalLayer() {}

    public ConvolutionalLayer(Network network, int layerIndex, int numFeatures, int featureDim) {
        this.network = network;
        this.layerIndex = layerIndex;
        this.convolutedDim = (network.HEIGHT - featureDim + 1);

        features = new ArrayList<>();
        for (int i=0; i<numFeatures; i++) {
            features.add(new FeatureMap(this, featureDim));
        }

    }

    @Override
    public void feedForward(List<Double> input, boolean backprop) {
        this.input  = input;
        List<Double> output = new ArrayList<>();

        for (FeatureMap map : features) {
            output.addAll(pool(map.filter(input)));
        }

        network.getLayer(layerIndex - 1).feedForward(output, backprop);
    }

    @Override
    public void backPropagate(List<Double> E) {

        for (int f=0; f<features.size(); f++) {

            int convolutedSize = (convolutedDim * convolutedDim) / 4;

            int index = f * convolutedSize;

            features.get(f).correct(E.subList(index, index + convolutedSize));

        }
    }

    private List<Double> pool(List<Double> input) {
        List<Double> out = new ArrayList<>();
        double o;

        for (int y = 0; y < convolutedDim - 2; y+= 2) {
            for (int x = 0; x < convolutedDim - 2; x+= 2) {

                o = Math.max(
                        input.get(y*convolutedDim + x), Math.max(
                        input.get(y*convolutedDim + x + 1), Math.max(
                        input.get((y+1)*convolutedDim + x), input.get((y+1)*convolutedDim + x + 1)
                )));

                out.add(o);
            }
        }
        return out;
    }

}
