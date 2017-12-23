module gru;

import activations : TanH, FastSigmoid;
import std.algorithm;

struct GRU (int neurons, int neuronsLayerBefore, DataType = float, alias activation = TanH!DataType, alias sigmoids = FastSigmoid!DataType) {
    static assert (neurons > 0 && neuronsLayerBefore > 0);
    alias WeightVector     = DataType [neuronsLayerBefore][neurons];
    alias LastWeightVector = DataType [neurons][neurons];
    alias OutVector        = DataType [neurons];
    alias InVector         = DataType [neuronsLayerBefore];

    WeightVector     weights;
    WeightVector     resetWeights;
    WeightVector     updateWeights;

    LastWeightVector lastWeights;
    LastWeightVector lastResetWeights;
    LastWeightVector lastUpdateWeights;

    OutVector        weightBiases;
    OutVector        resetBiases;
    OutVector        updateBiases;
    OutVector        hiddenState          = 0; // Last output.

    // TODO: Make optional.
    // Saves the sum of all the hidden state inputs.
    OutVector        hiddenStateSum       = 0;
    OutVector        candidateStateSum    = 0;
    OutVector        preCandidateStateSum = 0;
    OutVector        updateStateNegSum    = 0;
    OutVector        preUpdateStateSum    = 0;
    OutVector        weightsByHiddenSum   = 0;
    OutVector        resetStateSum        = 0;
    OutVector        preResetStateSum     = 0;
    InVector         inputSum             = 0;
    uint             forwardCalls         = 0;

    this (T) (T weightInitialization) {
        foreach (i; 0..neurons) {
            weights           [i][] = weightInitialization;
            lastWeights       [i][] = weightInitialization;
            resetWeights      [i][] = weightInitialization;
            lastResetWeights  [i][] = weightInitialization;
            updateWeights     [i][] = weightInitialization;
            lastUpdateWeights [i][] = weightInitialization;
        }
        weightBiases [] = weightInitialization;
        resetBiases  [] = weightInitialization;
        updateBiases [] = weightInitialization;
    }

    void reset () {
        hiddenState [] = 0;
    }

    // TODO: Don't save the sums if not training.
    // There seem to be several versions. This does Hadamard product of
    // the reset gate with U*h_{t-1} instead of U*(r hadamard h_{t-1}).
    void forward (in InVector input, out OutVector ret) {
        inputSum       [] += input       [];
        hiddenStateSum [] += hiddenState [];
        forwardCalls ++;
        foreach (i; 0..neurons) {
            import std.numeric : dotProduct;
            import activations : Sigmoid, TanH;

            /+
            // Tested and was too slow. :(
            auto preSum (in WeightVector weigths_, in WeightVector lastWeights_
                , in OutVector biasWeights) {

                return dotProduct (input, weigths_ [i])
                    + dotProduct (hiddenState, lastWeights_ [i])
                    + biasWeights [i];
            }+/
            // Reset calculation
            auto preReset = 
                dotProduct (input, resetWeights [i])
                + dotProduct (hiddenState, lastResetWeights [i])
                + resetBiases [i];

            preResetStateSum [i] += preReset;
            auto resetGate = sigmoids (preReset);
            resetStateSum [i] += resetGate;

            // Update calculation
            auto preUpdate =
                dotProduct (input, updateWeights [i])
                + dotProduct (hiddenState, lastUpdateWeights [i])
                + updateBiases [i];
            
            preUpdateStateSum [i] += preUpdate; 
            auto updateGate = sigmoids (preUpdate);
            updateStateNegSum [i] -= updateGate;

            auto weightByHidden = dotProduct (hiddenState, lastWeights [i]);
            weightsByHiddenSum [i] += weightByHidden;
            auto preCandidateState =
                dotProduct (input, weights [i]) 
                + resetGate * weightByHidden
                + weightBiases [i];

            preCandidateStateSum [i] += preCandidateState;
            auto h = activation (preCandidateState);
            
            candidateStateSum [i] += h;
            // New hidden state calculation
            hiddenState [i] = updateGate * hiddenState [i] + (1-updateGate) * h; 
            ret [i] = hiddenState [i];
        }
    }
    // errorVector contains the expected change in the outputs of this layer.
    // To determine how much to change for each neuron, the following algorithm
    // is used:
    // TODO: Might be useful to use doubles instead of floats for backprop.
    // TODO: Might be useful to separate the errorVector calculation into
    // another function so that neural.d can omit the calculation for the first layer.
    // Needs batchSize to calculate the means.
    void backprop (alias updateFunction) (
            in OutVector errorVector,
            out InVector errorGradientLayerBefore
        ) {
        /+debug {
            import std.stdio;
            writeln (`Biases before: `, weightBiases);
            writeln (`Weights before: `, weights);
        }+/
        errorGradientLayerBefore [] = 0;
        import std.functional : unaryFun;
        
        // Now they're averages.
        hiddenStateSum        [] /= forwardCalls;
        inputSum              [] /= forwardCalls;
        candidateStateSum     [] /= forwardCalls;
        updateStateNegSum     [] /= forwardCalls;
        weightsByHiddenSum    [] /= forwardCalls;
        resetStateSum         [] /= forwardCalls;
        preResetStateSum      [] /= forwardCalls;
        preUpdateStateSum     [] /= forwardCalls;
        preCandidateStateSum  [] /= forwardCalls;

        alias changeBasedOn = unaryFun!updateFunction;
        foreach (neuronPos, error; errorVector) {
            // TODO: Check if ignoring the update derivative helps on the long run.
            // Includes the sigmoids derivative for convenience.
            auto updateDerivative = error
                * (hiddenStateSum [neuronPos] - candidateStateSum [neuronPos])
                * sigmoids.derivative (preUpdateStateSum [neuronPos]);

            updateBiases [neuronPos] -= changeBasedOn (updateDerivative);

            // Update gate gradients.
            foreach (i, ref weight; updateWeights [neuronPos]) {
                errorGradientLayerBefore [i] += updateDerivative * weight;
                auto weightDerivative = updateDerivative * inputSum [i];
                weight -= changeBasedOn (weightDerivative);
            }
            foreach (i, ref weight; lastUpdateWeights [neuronPos]) {
                auto weightDerivative = updateDerivative * hiddenStateSum [i];
                weight -= changeBasedOn (weightDerivative);
            }

            auto candidateDerivative =
                error 
                * (updateStateNegSum [neuronPos] + 1)
                * activation.derivative (preCandidateStateSum [neuronPos]);

            weightBiases [neuronPos] -= changeBasedOn (candidateDerivative);

            foreach (i, ref weight; weights [neuronPos]) {
                errorGradientLayerBefore [i] += candidateDerivative * weight;
                auto weightDerivative = candidateDerivative * inputSum [i];
                weight -= changeBasedOn (weightDerivative);
            }

            foreach (i, ref weight; lastWeights [neuronPos]) {
                auto weightDerivative = candidateDerivative * hiddenStateSum [i];
                weight -= changeBasedOn (weightDerivative);
            }

            auto resetDerivative = 
                candidateDerivative 
                * weightsByHiddenSum [neuronPos]
                * sigmoids.derivative (preResetStateSum [neuronPos]);

            resetBiases [neuronPos] -= changeBasedOn (resetDerivative);
            foreach (i, ref weight; resetWeights [neuronPos]) {
                errorGradientLayerBefore [i] += resetDerivative * weight;
                auto weightDerivative = resetDerivative * inputSum [i];
                weight -= changeBasedOn (weightDerivative);
            }
            foreach (i, ref weight; lastResetWeights [neuronPos]) {
                auto weightDerivative = resetDerivative * hiddenStateSum [i];
                weight -= changeBasedOn (weightDerivative);
            }
        }

        // Resets so that next iteration can start filling.
        hiddenStateSum        [] = 0;
        inputSum              [] = 0;
        candidateStateSum     [] = 0;
        weightsByHiddenSum    [] = 0;
        preResetStateSum      [] = 0;
        preUpdateStateSum     [] = 0;
        preCandidateStateSum  [] = 0;
        forwardCalls             = 0;
        /+debug {
            import std.stdio;
            writeln (`Biases after: `, weightBiases);
            writeln (`Weights after: `, weights);
            writeln (`Activation errors: `, errorGradientLayerBefore);
        }+/
    }
}
