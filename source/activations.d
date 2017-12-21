module activations;

@safe @nogc pure nothrow {
    // Initializations
    auto ones () { return 1;}
    auto zeros () {return 0;} 


    // Activations.
    struct Linear (T = float) {
        static T opCall (T input) {return input;}
        static T derivative (T input) {return 1;}
    }
    struct ReLU (T = float) {
        static T opCall (T input) {return input > 0? input : 0;}
        static T derivative (T input) {return input > 0? 1 : 0;}
    }
    struct LeakyReLU (alias ratio, T = typeof (ratio)) {
        static T opCall (T input) {return input >= 0? input : input * ratio;}
        static T derivative (T input) {return input > 0? 1 : ratio;}
    }
    import std.math : log, exp, tanh, cosh, abs;
    struct TanH (T = float) {
        static T opCall (T input) {return tanh (input);}
        static T derivative (T input) {return 1/(cosh(input)*cosh(input));}
    }
    // TODO: Optimize this and FastSigmoid.
    // Might be faster to don't recalculate.
    struct Sigmoid (T = float) {
        static T opCall (T input) {return 1 / (1 + exp (-input));}
        static T derivative (T input) {
            return exp (input) / ((1 + exp (input)) * (1 + exp (input))); 
        }
    }
    // Elliot sigmoid.
    struct FastSigmoid (T = float) {
        static T opCall (T input) {return input / (1 + abs (input));}
        static T derivative (T input) {
            return input / ((abs (input) + 1) * (abs (input) + 1));
        }
    }
    struct Softplus (T = float) {
        static T opCall (T input) {return log (1 + exp (input));}
        static T derivative (T input) {return exp(input) / (exp(input) + 1);}
    }
    struct ELU (alias alpha, T = typeof (alpha)) {
        static T opCall (T input) {return input >= 0 ? input : alpha * (exp (input) -1);}
        static T derivative (T input) {return input >= 0 ? 1 : alpha * exp (input); }
    }
    struct SeLU (T = float) {
        enum alpha = 1.6732632423543772848170429916717;
        enum scale = 1.0507009873554804934193349852946;
        static T opCall (T input) {return scale * ELU!alpha (input);}
        static T derivative (T input) {return scale * ELU!alpha.derivative (input); }

    }
}
