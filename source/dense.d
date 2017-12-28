module dense;

import activations : Linear;
import std.algorithm;

import local;
template Dense (int neurons, int neuronsLayerBefore, DataType = float
    , alias activation = Linear!DataType) {
    alias Dense = Local !(neurons, neuronsLayerBefore, DataType, activation, 0);
}

unittest {
    // Manual neural network example.

    auto inputLayer = Dense! (4, 2) (0.2);
    inputLayer.weights = [[.1f,.2], [.3,.4], [.5,.6], [.7,.8]];
    inputLayer.biases   = [.1f, .2, .3, .4];
    auto hiddenLayer = Dense!(4, 4) (0.2);
    hiddenLayer.weights = [[.1f,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4]];
    hiddenLayer.biases   = [.1f, .2, .3, .4];
    auto outputLayer = Dense!(2, 4) (0.2);
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
    import optimizer;
    import std.conv;
    auto outputLayerOpt = RMSProp! (0.003, typeof (outputLayer), 0.0001, 0.5)(1);
    outputLayer.backprop (error, backpropBuffer, outputLayerOpt);
    /+
    hiddenLayer.backprop!`a/30`(backpropBuffer, backpropBuffer2);
    inputLayer.backprop!`a/30` (backpropBuffer2, cast (float [2]) backpropBuffer [0..2]);
    +/
}
