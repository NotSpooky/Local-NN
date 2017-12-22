module local;

import std.conv : text;

struct Local (int neurons, int neuronsLayerBefore, DataType = float
    , alias activation = Linear!DataType, int stride = 1) {

    static assert (neurons > 0 && neuronsLayerBefore > 0 && stride >= 0);
    enum connectivity = neuronsLayerBefore - (neurons * stride);
    static assert (connectivity > 0
        , `Local layer would have negative connectivity.`
    );

    static if (stride > connectivity) {
        pragma (msg, `Warning: stride for local layer is very high, `
            ~ `some neurons won't be connected`
        );
    }

    alias InVector     = DataType [neuronsLayerBefore];
    alias OutVector    = DataType [neurons];
    alias WeightVector = DataType [connectivity][neurons];

    WeightVector weights;
    OutVector    biases;
    InVector     activationSum          = 0;
    OutVector    preActivationOutputSum = 0;
    uint         forwardCalls           = 0;

    this (T) (T weightInitialization) {
        foreach (ref neuronWeights; weights) {
            neuronWeights [] = weightInitialization;
        }
        biases = weightInitialization;
    }

    void forward (in InVector lastLayerActivations, out OutVector ret) {
        activationSum [] += lastLayerActivations [];
        forwardCalls ++;
        foreach (i; 0..neurons) {
            import std.numeric : dotProduct;
            DataType preActivation = dotProduct (
                lastLayerActivations [i * stride .. i * stride + connectivity]
                , weights [i]
            ) + biases [i];
            preActivationOutputSum [i] += preActivation;
            ret [i] = activation (preActivation);
        }
    }

    void backprop (alias updateFunction) (
        in OutVector errorVector,
        out InVector errorGradientLayerBefore
    ) {
        activationSum            [] /= forwardCalls;
        preActivationOutputSum   [] /= forwardCalls;
        errorGradientLayerBefore [] = 0;
        import std.functional : unaryFun;
        alias changeBasedOn = unaryFun!updateFunction;
        foreach (neuronPos, error; errorVector) {
            auto effectInError = error 
                * activation.derivative (preActivationOutputSum [neuronPos]);
            biases [neuronPos] -= changeBasedOn (effectInError);
            foreach (j, weight; weights [neuronPos]) {
                errorGradientLayerBefore [neuronPos * stride + j] += effectInError * weight;
                auto weightDerivative = effectInError * activationSum [j];
                weight -= changeBasedOn (weightDerivative);
            }
        }
        activationSum          [] = 0;
        preActivationOutputSum [] = 0;
        forwardCalls              = 0;
    }
}
