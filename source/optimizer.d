enum trainable; // Used for UDAs

template Optimizer (alias Type, double learningRate, RestOfParameters ...) {
    // Useful so that Optimizer with the learning rate and type can be specified
    // to the neural network and the neural network uses this to instanciate
    // for each layer.
    alias optimizer (alias Layer) = Type! (learningRate, Layer, RestOfParameters);
}

// Optimizers must have a constructor that takes a non-used bool
// (structs cannot have 0-parameter constructors) :(

// TODO: Check if it's more useful to use doubles instead of DataType for floats.
struct Momentum (double learningRate, alias Layer, double momentumPercent = 0.7) {
    // Build members of the same type and length of every @trainable attribute 
    // of Layer.
    import std.traits;
    alias mems = getSymbolsByUDA! (Layer, trainable);
    static foreach (varName; mems) {
        mixin (`typeof (varName) ` ~ varName.stringof ~ `;`);
    }
    this (bool) {
        // Initialize with initVal instead of NaN.
        static foreach (varName; mems) {
            mixin (varName.stringof ~ `.recursiveInit (0);`);
        }
    }

    void setWeights (string memberName, D, Indices ...)
        (ref D currentValue, D gradient, Indices indices) {
        static assert (indices.length > 0, `setWeights needs indices.`);
        enum momentum = indexedRecursively! (memberName, Indices);
        // Eg. weights [4][3] = currentValue;
        auto change = ((1 - momentumPercent) * gradient 
                + momentumPercent * mixin (momentum));
        debug (2) {
            import std.stdio;
            write (memberName, ` `);
            write (`momentum = `, mixin (momentum), ` `);
            writeln (`change = `, change);
        }
        mixin (momentum) = change;
        currentValue -= learningRate * change;
    }
}


// epsilon is a small number to prevent division by 0.
struct RMSProp (double learningRate, alias Layer, double geometricRate = 0.9
    , double epsilon = 1e-10) {

    // Build members of the same type and length of every @trainable attribute 
    // of Layer.

    // Each contains the symbols of Layer that will be used here to create
    // variables with the same names.
    alias mems = getSymbolsByUDA! (Layer, trainable);
    import std.traits;
    static foreach (varName; mems) {
        mixin (`typeof (varName) ` ~ varName.stringof ~ `;`);
    }
    @disable this ();
    this (bool) {
        // Initialize with initVal instead of NaN.
        static foreach (varName; mems) {
            mixin (varName.stringof ~ `.recursiveInit (1);`);
        }
    }

    void setWeights (string memberName, D, Indices ...)
        (ref D currentValue, D gradient, Indices indices) {

        static assert (indices.length > 0, `setWeights needs indices.`);
        enum stored = indexedRecursively! (memberName, Indices);
        // Eg. weights [4][3] = currentValue;
        auto rms = geometricRate * mixin (stored)
            + (1 - geometricRate) * gradient * gradient;

        debug (2) {
            import std.stdio;
            writeln (stored);
            writeln (`currentValue = `, currentValue);
            writeln (`gradient = `, gradient);
            writeln (`stored = `, mixin (stored));
            writeln (`rms = `, rms);
        }
        mixin (stored) = rms;
        import std.math : sqrt;
        currentValue -= learningRate * gradient / sqrt (rms + epsilon);
        debug (2) {
            writeln (`Setting currentValue to: `, learningRate, ` * `
                , gradient, `/ sqrt (`, rms, ` + `, epsilon, `)`);
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
