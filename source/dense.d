module dense;

import activations : Linear;
import std.algorithm;

struct Dense (int neurons, int neuronsLayerBefore, DataType = float, alias activation = Linear!DataType) {
    static assert (neurons > 0 && neuronsLayerBefore > 0);
    alias WeightVector = DataType [neuronsLayerBefore][neurons];
    alias OutVector = DataType [neurons];
    alias InVector = DataType [neuronsLayerBefore];

    WeightVector weights;
    OutVector biases;
    
    this (T) (T weightInitialization) {
        foreach (ref neuronWeights; weights) {
            neuronWeights [] = weightInitialization;
        }
        biases [] = weightInitialization;
    }


    void forward (in InVector lastLayerActivations, out OutVector ret) {
        foreach (i; 0..neurons) {
            import std.numeric : dotProduct;
            ret [i] = activation (
                dotProduct (lastLayerActivations, weights [i]) + biases [i]
            );
        }
    }
    // errorVector contains the expected change in the outputs of this layer.
    // To determine how much to change for each neuron, the following algorithm
    // is used:
    // TODO: Might be useful to use doubles instead of floats for backprop.
    // TODO: Might be useful to separate the errorVector calculation into
    // another function so that neural.d can omit the calculation for the first layer.
    void backprop (alias updateFunction) (
        in OutVector errorVector,
        in InVector activationVector,
        out InVector errorGradientLayerBefore
    ) {
        // Implementation note: Weights and activations should be updated
        // simultaneously.
        /+debug {
            import std.stdio;
            writeln (`Biases before: `, biases);
            writeln (`Weights before: `, weights);
        }+/
        errorGradientLayerBefore [] = 0;
        import std.functional : unaryFun;
        alias changeBasedOn = unaryFun!updateFunction;
        foreach (neuronPos, error; errorVector) {
            auto effectInError = error * activation.derivative (error);
            biases [neuronPos] -= changeBasedOn (effectInError);
            foreach (j, weight; weights [neuronPos]) {
                errorGradientLayerBefore [j] += effectInError * weight;
                auto weightDerivative = effectInError * activationVector [j];
                weight -= changeBasedOn (weightDerivative);
            }
        }
        /+debug {
            import std.stdio;
            writeln (`Biases after: `, biases);
            writeln (`Weights after: `, weights);
            writeln (`Activation errors: `, errorGradientLayerBefore);
        }+/
    }
}

unittest {
    // Manual neural network example.

    auto inputLayer = Dense! (4, 2) ();
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
    float [4] backpropBuffer;
    float [4] backpropBuffer2;

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

    import std.conv;
    outputLayer.backprop!`a/30`(error, hiddenLayerOut, backpropBuffer);
    hiddenLayer.backprop!`a/30`(backpropBuffer, inputLayerOut, backpropBuffer2);
    inputLayer.backprop!`a/30` (backpropBuffer2, inputData, cast (float [2]) backpropBuffer [0..2]);
}
