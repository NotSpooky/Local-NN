module dense;

import activations;

struct Dense (int neurons, int neuronsLayerBefore, DataType = float, alias activation = Linear!DataType) {
    alias WeightVector = DataType [neuronsLayerBefore][neurons];
    alias OutVector = DataType [neurons];
    alias InVector = DataType [neuronsLayerBefore];
    WeightVector weights;
    OutVector biases;
    auto forward (in InVector lastLayerActivations, out OutVector ret) {
        foreach (i; 0..neurons) {
            import std.numeric : dotProduct;
            ret [i] = activation (dotProduct (lastLayerActivations, weights [i]) + biases [i]);
        }
    }
    // errorVector contains the expected change in the outputs of this layer.
    // To determine how much to change for each neuron, the following algorithm
    // is used:
    // TODO: Might be useful to use doubles instead of floats for backprop.
    import std.functional : unaryFun;
    auto backprop (alias updateFunction) (
        in OutVector errorVector,
        in InVector activations
    ) {

        /+debug {
            import std.stdio;
            writeln (`Biases before: `, biases);
            writeln (`Weights before: `, weights);
        }+/
        alias changeBasedOn = unaryFun!updateFunction;
        DataType [neuronsLayerBefore] activationErrors = 0;
        foreach (neuronPos, error; errorVector) {
            auto effectInError = error * activation.derivative (error);
            biases [neuronPos] -= changeBasedOn (effectInError);
            foreach (j, ref weight; weights [neuronPos]) {
                activationErrors [j] += effectInError * weight;
                auto weightDerivative = effectInError * activations [j];
                weight -= changeBasedOn (weightDerivative);
            }
        }
        /+debug {
            import std.stdio;
            writeln (`Biases after: `, biases);
            writeln (`Weights after: `, weights);
            writeln (`Activation errors: `, activationErrors);
        }+/
        return activationErrors;
    }
}

/// Expects layers to have the following CT parameter format:
/// neurons, neuronsLayerBefore, rest of parameters.

/+
// TODO: Allow arrays of layers or similar.
/// A succession of numLayers identical dense layers.
struct NeuralNetwork (DataType, uint inputLen, alias activation, Layers ...) {
    static foreach (i, layer; Layers) {
        mixin (text(layer.stringof ~ `!(` ~ `)` ~ ` layer`, i, ` `
        pragma (msg, i);
        pragma (msg, layer!(3,4));
    }
}
+/

private struct LayerData {
    string type;
    uint neurons;
    /// Must be comma-separated and in the order that they are inserted.
    string parameters; 
    
}

private string nnGenerator (int inputLen, LayerData [] layers) {
    assert (inputLen > 0);
    assert (layers.length);
    import std.array : Appender;
    import std.conv : text;
    Appender!string toReturn;
    foreach (i, layer; layers) {
        auto layerInputLen = i == 0 ? inputLen : layers [i-1].neurons;
        toReturn ~= text (
            // Eg. Dense
            layer.type 
            //Eg. !(3, 4, float, Linear!float)
            , `!(`    , layer.neurons
                , `, `, layerInputLen
                , `, `, layer.parameters
            ,`)` 
            // Eg. layer3;
            , ` layer`, i , `;`);
    }
    return toReturn.data;
}

unittest {

    //import std.meta : AliasSeq;
    //NeuralNetwork!(float, 2, 2, Linear!float, Dense) nn;
    debug {
        import std.stdio;
        writeln (nnGenerator (4, [LayerData (`Dense`, 8, `float, Linear!float`)]));
    }

    auto inputLayer = Dense!(4, 2) ();
    inputLayer.weights = [[.1f,.2], [.3,.4], [.5,.6], [.7,.8]];
    inputLayer.biases   = [.1f, .2, .3, .4];
    auto hiddenLayer = Dense!(4, 4) ();
    hiddenLayer.weights = [[.1f,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4]];
    hiddenLayer.biases   = [.1f, .2, .3, .4];
    auto outputLayer = Dense!(2, 4) ();
    outputLayer.weights = [[.1f,.2,.3,.4], [.1f,.2,.3,.4]];
    outputLayer.biases   = [.1f, .2];

    float [2] inputData = [0f, 4f];
    float [2] expectedOutput = [2f, 3];
    float [2] outputLayerOut;

    import std.stdio;
    float [4] inputLayerOut;
    inputLayer.forward (inputData, inputLayerOut);
    //writeln (`inputLayerOut: `, inputLayerOut);

    float [4] hiddenLayerOut;
    hiddenLayer.forward (inputLayerOut, hiddenLayerOut);
    //writeln (`hiddenLayerOut: `, hiddenLayerOut);

    outputLayer.forward (hiddenLayerOut, outputLayerOut);
    //writeln (`outputLayerOut: `, outputLayerOut);

    float [2] error = outputLayerOut [] - expectedOutput [];
    //writeln (`Linear error: `, error);

    auto hiddenErrors = outputLayer.backprop!`a/30`(error, hiddenLayerOut);
    auto inputErrors =  hiddenLayer.backprop!`a/30`(hiddenErrors, inputLayerOut);
    inputLayer.backprop!`a/30` (inputErrors, inputData);
}
