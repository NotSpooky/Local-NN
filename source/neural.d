module neural;

import std.functional : unaryFun, binaryFun;
import std.algorithm;
import std.conv  : text, to;
static import std.file;

// TODO: Allow arrays of layers or similar.
// TODO: Error function gradient.
// TODO: Use the predictions instead of average output.


private auto getValue (alias value) () {
    import std.traits : isCallable;
    static if (isCallable!value) {
        return value ();
    } else {
        return value;
    }
}

/// Expects layers to have the following CT parameter format:
/// neurons, neuronsLayerBefore, DataType, activation, rest of parameters.
// Layers are named layer0, layer1 ...
final class NeuralNetwork (int inputLen, DataType, alias weightInitialization, layers ...) {
    import std.range;
    mixin (nnGenerator (inputLen, layers));

    this () {
        static foreach (i, layer; layers) {
            // Call the constructors of all the layers.
            mixin (text (
                `layer`, i, ` = typeof (layer`, i, `)`
                ~ `(getValue!weightInitialization);`
            ));
        }
    }

    

    // Version that returns the array by the output parameter.
    // Allows multidimensional outputs.
    auto predict (R, RO) (in R [inputLen] input, out RO [outputLen] output) {
        forward!false (input, output);
    }

    // Version that creates a new array and returns it.
    // Assumes that the output has one dimension.
    auto predict (R) (in R [inputLen] input) {
        //auto toReturn = new DataType [outputLen];
        DataType [outputLen] toReturn;
        predict (input, toReturn);
        return toReturn.dup;
    }

    private void forward 
        (
            bool training = false, SingleInputType
        ) (
            in SingleInputType [inputLen] input
            , out SingleInputType [outputLen] output
        ) {

        // Used to allocate an array of the appropiate length for the buffers.
        // Could be optimized to use the second biggest also for the other buffer.
        // uint maxAmountOfNeurons = layers.map!(a => a.neurons).reduce!max;
        enum maxAmountOfNeurons = () {
            uint toRet = 0;
            foreach (layer; layers) {
                toRet = max (toRet, layer.neurons);
            }
            return toRet;
        } ();

        // TODO: Now one less buffer is needed.
        static if (layers.length > 1) {
            DataType [maxAmountOfNeurons][2] buffers;
        } else {
            DataType [maxAmountOfNeurons][1] buffers;
        }

        //pragma (msg, generateForward (training));
        mixin (generateForward (training));
    }

    // Used to allocate an array for backpropagation gradients.
    // TODO: If the backprop algorithm stops writing the last
    // buffer. Could reduce the size by not considering inputLen.
    enum biggestInput = () {
        uint toRet = inputLen;
        foreach (layer; layers [0..$-1]) {
            toRet = max (toRet, layer.neurons);
        }
        return toRet;
    } ();

    import optimizer : Optimizer;
    // TODO: Maybe eliminate the need for another array if the error
    // function doesn't need it (for example mse).
    // TODO: Add Random access range constraints.
    void train (
        alias optimizer, 
        alias errorFunction, 
        bool printError   = true,
        bool shuffleInput = true,
        bool doInParallel = true
        , R1, R2
    ) (int epochs, int batchSize, R1 inputs, R2 labels) {

        assert (inputs.length == labels.length);
        static if (hasStatefulLayers) {
            static assert (
                is (typeof (inputs.front.front.front) == DataType)
                , `Network is stateful, must receive sequences`
            );

            // Technically should test the entire dataset, but that's too
            // expensive.
            assert (
                inputs.front.front.length == inputLen
                , `incorrect input length for training.`
            );
        } else {
            static assert (is (typeof (inputs.front.front) == DataType), 
                text (`Incorrect data type for input, should be `
                    , DataType.stringof
                )
            );
            // Technically should test the entire dataset, but that's too
            // expensive.
            assert (
                inputs.front.length == inputLen
                , `incorrect input length for training.`
            );
        }

        // TODO: Allow multidimensional outputs.
        static assert (is (typeof (labels.front.front) == DataType), 
            text (`Incorrect data type for labels, should be `
                , DataType.stringof
            )
        );
        static assert (__traits(compiles, optimizer.optimizer!(typeof(layer0)))
            , `First compile-time parameter of train should be an Optimizer.`);

        assert (labels.front.length == outputLen
            , `incorrect output length for training`
        );

        static foreach (i, layer; layers) {
            // Eg. auto optimizer2 = RMSProp!(0.001, typeof (layer2))(true);
            mixin (text (`auto optimizer`, i
                , q{ = optimizer.optimizer!(typeof (layer}, i, q{) )(true);})
            );
        }

        foreach (epoch; 0..epochs) {
            auto indices = iota (inputs.length).array;
            static if (shuffleInput) {
                // TODO: Maybe is faster if indices are ommited (zip and shuffle)
                import std.random : randomShuffle;
                indices.randomShuffle;
            }
            static if (printError) {
                DataType errorToPrint = 0;
                DataType [outputLen] outputError = 0;
            }
            foreach (batch; indices.chunks (batchSize)) {
                auto dataChunks  = inputs.indexed (batch);
                auto labelChunks = labels.indexed (batch);
                DataType [outputLen   ] averageOutputError = 0;
                DataType [outputLen   ] output;

                static if (doInParallel) {
                    import std.parallelism;
                    alias fun = (a) => parallel (a);
                    static assert (0, `TODO: Fix race conditions`);
                } else {
                    alias fun = (a) => a;
                }
                foreach (exampleLabel; fun (zip (dataChunks, labelChunks))) {
                    static if (hasStatefulLayers) {
                        // Each example is divided into steps.
                        foreach (step; exampleLabel [0]) {
                            this.forward!true (
                                step.to!(DataType [inputLen]),
                                output
                            );
                        }
                        this.reset;
                    } else {
                        // Just one pass per example.
                        this.forward!true (
                            exampleLabel [0].to!(DataType [inputLen]),
                            output,
                        );
                    }
                    binaryFun!errorFunction (
                        output, exampleLabel [1], averageOutputError
                    );
                    static if (printError) {
                        binaryFun!errorFunction (
                            output, exampleLabel [1], outputError
                        );
                        import std.math  : abs;
                        errorToPrint += outputError [].map!abs.sum;
                        outputError [] = 0;
                    }
                }
                averageOutputError [] /= batch.length;
                // Gradient of layer activations w.r.t. layer output error.
                DataType [biggestInput] [2] backError; 
                mixin (mixBackprop ());
            }
            static if (printError) {
                import std.stdio : writeln;
                writeln (`Average error = `
                    , errorToPrint / (outputLen * inputs.length));
                errorToPrint = 0;
            }
        }
    } // End of train

    // Note: this writes the file in several steps. Make sure there's not
    // another process/thread writing to it.
    void serialize (string filename) {
        // Clear the file.
        std.file.write (filename, []); 
        // Append to it.
        foreach (i, layer; layers) {
            serializeLayer (filename, mixin (text (`layer`, i)));
        }
    }
    private static string mixBackprop () {
        import std.array : Appender;
        Appender!string toReturn;
        foreach_reverse (i, layer; layers) {
            static if (i == layers.length - 1) {
                string errorIn = `averageOutputError`;
            } else {
                string errorIn = text (
                    `backError [`, (i + 1) % 2, `][0..`, layer.neurons ,`]`
                );
            }
            static if (i == 0) {
                // First layer. Really not necessary to store this.
                string gradientOutput = text (
                    `cast (DataType [inputLen]) backError [0][0..inputLen]`
                );
            } else {
                uint neuronsLBefore = layers [i - 1].neurons;
                string gradientOutput = text (
                    `cast (DataType [`, neuronsLBefore ,`])`
                    , ` backError [`, i % 2, `][0..`, neuronsLBefore ,`]`
                );
            }
            toReturn ~= text(
                `layer`, i, `.backprop (`
                    , errorIn, `, `, gradientOutput, `, optimizer`, i
                , ");\n"
            );
        }
        return toReturn.data;
    }

    // TODO: Reset the internal state of the resetteable layers.
    private static string generateForward (bool training) {
    Appender!string toReturn;

    // Each layer outputs to alternating buffers.
    foreach (i, layer; layers) {
    static if (i == 0) {
    // First layer receives from input.
    string layerInput = `input`; 
    } else {
    // Other layers receive from the output of the last layer.
    string layerInput = text (
            `buffers [`, (i + 1) % 2, `][0..`,layers [i-1].neurons,`]`
            );
    }
    static if (i == layers.length - 1) {
        // Last layer outputs to output.
        string layerOutput = `output`;
    } else {
        string layerOutput = text (
                `buffers [`,i % 2, `][0..`, layer.neurons ,`]`
                );
    }

    // Eg. layer0.forward (input, cast (float [8]) buffers [0][0..8]);
    //  layer1.forward (
    //     buffers [0][0..8], cast (float [16]) buffers [1][0..16]
    //  );
    //  return buffers [1][0..16].dup;
    toReturn ~= text (
            `layer`, i, `.forward (`
                , layerInput 
                , `, cast (DataType [`, layer.neurons , `])`
                , layerOutput
                , ");\n"
                );
            }
            return toReturn.data;
            }

    private enum hasStatefulLayers = () {
        bool toReturn = false;
        foreach (i, layer; layers) {
            if (__traits(compiles, mixin (text (`layer`, i, `.reset`))))
                // Has reset method.
                toReturn = true;
        }
        return toReturn;
    } ();
    
    static if (hasStatefulLayers) {
        private void reset () {
            static foreach (i, layer; layers) {
                // Has reset method.
                static if (__traits(compiles, mixin (text (`layer`, i, `.reset`))))
                    mixin (text (`layer`, i, `.reset;`));
            }
        }
    }
}

import activations : Linear;
// TODO: Use AliasSeq instead of parameter string.
struct Layer (alias Type, alias Activation = Linear!float) {
    alias type = Type;
    alias activation = Activation;
    uint neurons;
    /// Must be comma-separated and in the order that they are inserted.
    string parameters; 
}

private string nnGenerator (Layers ...) (int inputLen, Layers layers) {
    assert (inputLen > 0);
    static assert (layers.length, `Cannot create empty neural network.`);
    import std.array : Appender;
    import std.meta  : staticMap;
    
    // Last layer doesn't need to store its activations for backpropagation.
    // uint activationNeurons = layers [0..$-1].map!(a => a.neurons).sum;
    uint activationNeurons = 0;
    foreach (layer; layers [0 .. $-1]) {
        activationNeurons += layer.neurons;
    }

    Appender!string toReturn;

    // Useful constants
    toReturn ~= text (
        `enum activationNeurons = `, activationNeurons, ";\n"
        , `enum inputLen = `, inputLen, ";\n"
        , `enum outputLen = `, layers [$ - 1].neurons, ";\n"
    );

    ////////////////////
    // Layer creation //
    ////////////////////

    foreach (i, layer; layers) {
        static if (i == 0) {
            auto layerInputLen = inputLen;
        } else {
            auto layerInputLen = layers [i - 1].neurons;
        }
        toReturn ~= text (
            // Eg. Dense
            `layers [`, i, `].type`
            //Eg. !(3, 4, float, Linear!float)
            , `!(`
                , layer.neurons, `, `
                , layerInputLen
                , `, DataType`
                , `, layers [`, i, `].activation`
                , `, `, layer.parameters
                ,`)` 
            // Eg. layer3;
            , ` layer`, i , ";\n"
        );
    }

    return toReturn.data;
}

import std.traits : getSymbolsByUDA;
import optimizer  : trainable;
private auto serializeLayer (Layer)(string filename, Layer toSerialize) {
    foreach (symbol; getSymbolsByUDA! (Layer, trainable)) {
        std.file.append
            (filename, __traits (getMember, toSerialize, symbol.stringof));
    }
}
private auto deserializeLayer (Layer)(string filename, out Layer output) {
    auto file = std.file.read (filename);
    foreach (symbol; getSymbolsByUDA! (Layer, trainable)) {
        auto member = cast (void []) __traits (getMember, output, symbol.stringof);
        member [0..member.length] = file [0..member.length];
        file = file [member.length .. $];

    }
}

unittest {
    import dense;
    import local;
    import gru;
    import activations;
    import optimizer;
    debug {
        auto a = new NeuralNetwork !
        (
             /* Input length */ 4
             , float
             , 0.2 // Can use both a function or a value as weight initialization.
             , Layer! (Dense) (8)
             , Layer! (Dense, ReLU!float) (16)
             , Layer! (GRU, TanH!float) (16)
             , Layer! (Local, LeakyReLU!0.2f) (4)
             , Layer! (Dense, Linear!float) (2)
        ); 

        float [2] output;
        float [4] input = [1,2,3,4];
        import std.stdio;
        //a.predict (input, output);
        //writeln (output);
        //a.predict (input).writeln;
        // Using just the derivative :)
        static void meanSquaredError (A, B, C)
            (A output, in B expected, ref C accumulated) {

            /+
            output [] = expected [] - output [];
            accumulated [] += output [] * output [];
            +/
            accumulated [] += output [] - expected [];
        }
        
        a.train! (
            Optimizer! (Momentum, 0.005, 0.3)
            , meanSquaredError
            , false /* No stdout*/
            , true
            , false
        )(
              1 /* Just 3 epochs for testing */
            , 2 /* Batch size */
            , [[[1f,2,3,4]],[[2f,3,4,5]],[[3f,4,5,6]]]
            , [[4f,3], [5f,4], [6f,5]]
        );
        a.serialize (`test.weights`);
        //a.predict (input).writeln;
    }
}
pragma (msg, `Might check the CPU type in the dflags section of dub.json -mcpu=skylake was slower than not writing anything`);
