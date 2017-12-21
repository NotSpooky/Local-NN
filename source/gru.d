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
    OutVector        hiddenState        = 0; // Last output.

    // TODO: Make optional.
    // Saves the sum of all the hidden state inputs.
    OutVector        hiddenStateSum     = 0;
    OutVector        candidateStateSum  = 0;
    OutVector        updateStateNegSum  = 0;
    OutVector        weightsByHiddenSum = 0;
    OutVector        resetStateSum      = 0;
    InVector         inputSum           = 0;
    uint             forwardCalls       = 0;

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
            // Reset calculation
            auto resetGate = sigmoids (
                dotProduct (hiddenState, lastResetWeights [i])
                + dotProduct (input, resetWeights [i]) + resetBiases [i]
            );
            resetStateSum [i] += resetGate;

            // Update calculation
            auto updateGate = sigmoids (
                dotProduct (hiddenState, lastUpdateWeights [i])
                + dotProduct (input, updateWeights [i]) + updateBiases [i]
            ); 
            updateStateNegSum [i] -= updateGate;

            auto weightByHidden = dotProduct (hiddenState, lastWeights [i]);
            weightsByHiddenSum [i] += weightByHidden;
            auto h = activation (
                dotProduct (input, weights [i]) 
                + resetGate * weightByHidden
                + weightBiases [i]
            );
            
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
            in InVector activationVector,
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
        hiddenStateSum     [] /= forwardCalls;
        inputSum           [] /= forwardCalls;
        candidateStateSum  [] /= forwardCalls;
        updateStateNegSum  [] /= forwardCalls;
        weightsByHiddenSum [] /= forwardCalls;
        resetStateSum      [] /= forwardCalls;

        alias changeBasedOn = unaryFun!updateFunction;
        foreach (neuronPos, error; errorVector) {
            // TODO: Check if ignoring the update derivative helps on the long run.
            // Includes the sigmoids derivative for convenience.
            auto updateDerivative = error
                * (hiddenStateSum [neuronPos] - candidateStateSum [neuronPos])
                * sigmoids.derivative (error);

            updateBiases [neuronPos] -= changeBasedOn (updateDerivative);

            foreach (i, ref weight; updateWeights [neuronPos]) {
                auto weightDerivative = updateDerivative * inputSum [i];
                weight -= changeBasedOn (weightDerivative);
            }
            foreach (i, ref weight; lastUpdateWeights [neuronPos]) {
                auto weightDerivative = updateDerivative * hiddenStateSum [i];
                weight -= changeBasedOn (weightDerivative);
            }

            auto candidateDerivative =
                error 
                * updateStateNegSum [neuronPos] 
                * activation.derivative (error);

            weightBiases [neuronPos] -= changeBasedOn (candidateDerivative);

            foreach (i, ref weight; weights [neuronPos]) {
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
                * sigmoids.derivative (candidateDerivative);

            resetBiases [neuronPos] -= changeBasedOn (resetDerivative);
            foreach (i, ref weight; resetWeights [neuronPos]) {
                auto weightDerivative = resetDerivative * inputSum [i];
                weight -= changeBasedOn (weightDerivative);
            }
            foreach (i, ref weight; lastResetWeights [neuronPos]) {
                auto weightDerivative = resetDerivative * hiddenStateSum [i];
                weight -= changeBasedOn (weightDerivative);
            }
            //static assert (0, `TODO: GRU backprop`);
            /+
            auto effectInError = error * activation.derivative (error);
            weightBiases [neuronPos] -= changeBasedOn (effectInError);
            foreach (j, weight; weights [neuronPos]) {
                errorGradientLayerBefore [j] += effectInError * weight;
                auto weightDerivative = effectInError * activationVector [j];
                weight -= changeBasedOn (weightDerivative);
            }
            +/
        }

        // Resets so that next iteration can start filling.
        hiddenStateSum     [] = 0;
        inputSum           [] = 0;
        candidateStateSum  [] = 0;
        weightsByHiddenSum [] = 0;
        forwardCalls          = 0;
        /+debug {
            import std.stdio;
            writeln (`Biases after: `, weightBiases);
            writeln (`Weights after: `, weights);
            writeln (`Activation errors: `, errorGradientLayerBefore);
        }+/
    }
}
