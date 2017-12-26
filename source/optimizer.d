enum trainable; // Used for UDAs

// TODO: Check if it's more useful to use doubles instead of DataType for floats.
struct Momentum (double learningRate, Layer, double momentumPercent) {
    // Build members of the same type and length of every @trainable attribute 
    // of Layer.
    import std.traits;
    static foreach (trainable; getSymbolsByUDA!(Layer, trainable)) {
        mixin (`typeof (trainable) ` ~ trainable.stringof ~ `= [0];`);
    }

    void setWeights (string memberName, D, Indices ...)
        (ref D currentValue, D gradient, Indices indices) {
        static assert (indices.length > 0, `setWeights needs indices.`);
        enum varToUse = indexedRecursively! (memberName, Indices);
        // Eg. weights [4][3] = currentValue;
        auto change = ((1 - momentumPercent) * gradient 
                + momentumPercent * mixin (varToUse));
        debug {
            import std.stdio;
            writeln (`change = `, change);
            writeln (`varToUse = `, mixin (varToUse));
        }
        mixin (varToUse) = change;
        currentValue -= learningRate * change;
    }
}


struct RMSProp (double learningRate, Layer, double iotaS, double geometricRate) {
    // Build members of the same type and length of every @trainable attribute 
    // of Layer.

    // Each contains the symbols of Layer that will be used here to create
    // variables with the same names.
    alias mems = getSymbolsByUDA!(Layer, trainable);
    import std.traits;
    static foreach (varName; mems) {
        mixin (`typeof (varName) ` ~ varName.stringof ~ `;`);
    }
    @disable this ();
    this (D)(D initVal) {
        // Initialize with initVal instead of NaN.
        static foreach (varName; mems) {
            mixin (varName.stringof ~ `.recursiveInit (initVal);`);
        }
    }

    void setWeights (string memberName, D, Indices ...)
        (ref D currentValue, D gradient, Indices indices) {
            static assert (indices.length > 0, `setWeights needs indices.`);
            enum stored = indexedRecursively! (memberName, Indices);
            // Eg. weights [4][3] = currentValue;
            auto change = (1 - geometricRate) * mixin (stored)
                + geometricRate * gradient * gradient;

            debug {
                import std.stdio;
                writeln (stored);
                writeln (`currentValue = `, currentValue);
                writeln (`gradient = `, gradient);
                writeln (`stored = `, mixin (stored));
                writeln (`change = `, change);
            }
            mixin (stored) = change;
            import std.math : sqrt;
            currentValue -= learningRate * gradient / sqrt (change + iotaS);
            debug {
                writeln (`Setting currentValue to: `, learningRate, ` * `
                    , gradient, `/ sqrt (`, change, ` + `, iotaS, `)`);
                writeln;
            }
        }
}

// Used so that members of different dimensions can be indexed.
// Assumes there's a variadic Indices template whose variable is named
// indices.
// Returns eg. `biases [indices [0]][indices[1]]`
private string indexedRecursively (string memberName, Indices ...)() {
    import std.algorithm;
    string indicesStr = ``;
    foreach (i, iType; Indices) {
        import std.conv : text;
        indicesStr ~= text (`[indices [`, i, `]]`);
    }
    return memberName ~ indicesStr; 
}

private void recursiveInit (ArrayType, DataType) (ref ArrayType arr, DataType initValue) {
    import std.range : front;
    static if (__traits (compiles, arr = initValue)) {
        arr = initValue;
    } else {
        foreach (ref val; arr) {
            recursiveInit (val, initValue);
        }
    }
    
}
