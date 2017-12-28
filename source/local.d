module local;

import std.conv : text;

struct Local (int neurons, int neuronsLayerBefore, DataType = float
    , alias activation = Linear!DataType, int stride = 1) {

    static assert (neurons > 0 && neuronsLayerBefore > 0 && stride >= 0);
    enum connectivity = neuronsLayerBefore - (neurons * stride);
    static assert (connectivity > 0
        , `Local layer would have non-positive connectivity.`
    );

    static if (stride > connectivity) {
        pragma (msg, `Warning: stride for local layer is very high, `
            ~ `some neurons won't be connected`
        );
    }

    alias InVector     = DataType [neuronsLayerBefore];
    alias OutVector    = DataType [neurons];
    alias WeightVector = DataType [connectivity][neurons];

    import optimizer : trainable;
    @trainable WeightVector weights;
    @trainable OutVector    biases;
               InVector     activationSum          = 0;
               OutVector    preActivationOutputSum = 0;
               uint         forwardCalls           = 0;

    this (T) (T weightInitialization) {
        foreach (ref neuronWeights; weights) {
            neuronWeights [] = weightInitialization;
        }
        biases = weightInitialization;
    }
    @disable this ();

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

    void backprop (Optimizer)(
        in OutVector errorVector,
        out InVector errorGradientLayerBefore,
        ref Optimizer optimizer
    ) {
        activationSum            [] /= forwardCalls;
        preActivationOutputSum   [] /= forwardCalls;
        errorGradientLayerBefore [] = 0;
        import std.functional : unaryFun;
        foreach (neuronPos, error; errorVector) {
            auto effectInError = error 
                * activation.derivative (preActivationOutputSum [neuronPos]);
            optimizer.setWeights!`biases` (biases [neuronPos], effectInError, neuronPos);
            foreach (j, weight; weights [neuronPos]) {
                errorGradientLayerBefore [neuronPos * stride + j] += effectInError * weight;
                auto weightDerivative = effectInError * activationSum [j];

                optimizer.setWeights!`weights` 
                    (weight, weightDerivative, neuronPos, j);
            }
        }
        activationSum          [] = 0;
        preActivationOutputSum [] = 0;
        forwardCalls              = 0;
    }
}

unittest {
    import activations;
    auto layer = Local! (2, 4, float, Linear!float) (0.2);
    typeof (layer.weights) testArray = 0;

}
