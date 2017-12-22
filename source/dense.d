module dense;

import activations : Linear;
import std.algorithm;

struct Dense (int neurons, int neuronsLayerBefore, DataType = float
    , alias activation = Linear!DataType) {

    static assert (neurons > 0 && neuronsLayerBefore > 0);
    alias WeightVector = DataType [neuronsLayerBefore][neurons];
    alias OutVector    = DataType [neurons];
    alias InVector     = DataType [neuronsLayerBefore];

    WeightVector weights;
    OutVector    biases;
    // TODO: Make optional.
    InVector     activationSum          = 0;
    OutVector    preActivationOutputSum = 0;
    uint         forwardCalls           = 0;
    
    this (T) (T weightInitialization) {
        foreach (ref neuronWeights; weights) {
            neuronWeights [] = weightInitialization;
        }
        biases [] = weightInitialization;
    }


    void forward (in InVector lastLayerActivations, out OutVector ret) {
        forwardCalls ++;
        activationSum [] += lastLayerActivations [];
        foreach (i; 0..neurons) {
            import std.numeric : dotProduct;
            DataType preActivation = dotProduct ( 
                lastLayerActivations, weights [i]
            ) + biases [i];
            preActivationOutputSum [i] += preActivation;
            ret [i] = activation (preActivation);
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
        out InVector errorGradientLayerBefore
    ) {
        // Implementation note: Weights and activations should be updated
        // simultaneously.
        /+debug {
            import std.stdio;
            writeln (`Biases before: `, biases);
            writeln (`Weights before: `, weights);
        }+/
        activationSum          [] /= forwardCalls;
        preActivationOutputSum [] /= forwardCalls;
        errorGradientLayerBefore [] = 0;
        import std.functional : unaryFun;
        alias changeBasedOn = unaryFun!updateFunction;
        foreach (neuronPos, error; errorVector) {
            auto effectInError = error 
                * activation.derivative (preActivationOutputSum [neuronPos]);
            biases [neuronPos] -= changeBasedOn (effectInError);
            foreach (j, weight; weights [neuronPos]) {
                errorGradientLayerBefore [j] += effectInError * weight;
                auto weightDerivative = effectInError * activationSum [j];
                weight -= changeBasedOn (weightDerivative);
            }
        }
        activationSum          [] = 0;
        preActivationOutputSum [] = 0;
        forwardCalls              = 0;
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
    outputLayer.backprop!`a/30`(error, backpropBuffer);
    hiddenLayer.backprop!`a/30`(backpropBuffer, backpropBuffer2);
    inputLayer.backprop!`a/30` (backpropBuffer2, cast (float [2]) backpropBuffer [0..2]);
}
