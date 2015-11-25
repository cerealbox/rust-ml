var println = console.log.bind(console)

function executeRange(start, end, fn) {
    for (let i = start; i < end; i++) {
        fn(i)
    }
}

Array.prototype.scan = function(fn, start) {
    let acc = start
    let results = []
    for (let i = 0; i < this.length; i++) {
        results.push(acc = fn(acc, this[i]))
    }
    return results
}

Array.prototype.takeWhile = function(fn) {
    let results = []
    for (let i = 0; i < this.length; i++) {
        if (fn(this[i]))
            results.push(i)
        else
            return results
    }
    return results
}

// Stuff I'd like to do:
// Most common layer types, easy addition of new types
// Fluent, iterator-like API
// Snapshots
// Serialization
// Larger-than-resident-memory nets
// SLI
// Cross-machine
// Heterogenous networks

function zero_vec(size) {
    var result = []
    for (var i = 0; i < size; i++) {
        result.push(0.0)
    }
    return result
}

function random_vec(size) {
    var result = []
    for (var i = 0; i < size; i++) {
        result.push(Math.random())
    }
    return result
}

function layer(size, input_size) {
    return {
        size: size,
        neurons: zero_vec(size),
        deltas: zero_vec(size),
        errors: zero_vec(size),
        biases: random_vec(size),
        weights: random_vec(size * input_size),
        changes: zero_vec(size * input_size),
        clone: () => layer(size, input_size)
    }
}

function BasicSolver(learning_rate, momentum) {
    this.target_output = [],
    this.learning_rate = learning_rate,
    this.momentum = momentum
}

BasicSolver.prototype.feed_forward = function(input_layer, output_layer) {
    let input_size = input_layer.length;
    let output_size = output_layer.length;
    let result = 0.0;

    executeRange(0, output_size, (i) => {
        let bias = output_layer.biases[i];
        let sum = input_layer.neurons.reduce( (sum, [j, input_neuron]) => {
            let weight_index = (input_size * i) + j;
            return sum + output_layer.weights[weight_index] * input_neuron
        }, bias);

        result = 1.0 / (1.0 + Math.exp(-sum));
        output_layer.neurons[i] = result;
    })

    return result
}

//BasicSolver.prototype.calculate_deltas(input_layer: Option<&RefCell<Layer>>, output_layer: &mut Layer) {
BasicSolver.prototype.calculate_deltas = function(input_layer, output_layer) {
    let output_size = output_layer.length;

    executeRange(0, output_size, (i) => {
        let neuron = output_layer.neurons[i];

        if (input_layer != null) {
            let layer = input_layer
            let error = layer.deltas.reduce((sum, [j, delta]) => {
                let weight_index = (output_size * j) + i;
                return sum + (delta * layer.weights[weight_index])
            }, 0.0)
        } else {
            let error = this.target_output[i] - neuron
        }

        output_layer.errors[i] = error;
        output_layer.deltas[i] = error * neuron * (1.0 - neuron);
    })
}

BasicSolver.prototype.adjust_weights = function(input_layer, output_layer) {
    let input_size = input_layer.length;
    let output_size = output_layer.length;
    let learning_rate = this.learning_rate;
    let momentum = this.momentum;

    executeRange(0, output_size, (i) => {
        let delta = output_layer.deltas[i];

        input_layer.neurons.forEach(([j, neuron]) => {
            let change_index = (input_size * i) + j;
            let change = output_layer.changes[change_index];

            change = (learning_rate * delta * neuron) + (momentum * change);

            output_layer.changes[change_index] = change;
            output_layer.weights[change_index] += change;
        })

        output_layer.biases[i] += learning_rate * delta;
    })
}


//    layers: Vec<RefCell<Layer>>
function Network(sizes) {

    this.solver = new BasicSolver(0.3, 0.1)
    this.layers = 
        sizes.
            scan((prev_size, size) => {
                let result = layer(size, prev_size)
                prev_size = size;
                return result
            }, 0).
            // map(|layer| RefCell::new(layer)).
            map((layer) => layer)
}

// Network.prototype.run = function(input: &Vec<f32>) -> f32 {
Network.prototype.run = function(input) {
        //TODO: Compare lengths

        let result = 0.0;
        let layers_len = this.layers.length;

        let input_layer = this.layers[0]
        input_layer.neurons = input

        executeRange(1, layers_len, (i) => {
            let input_layer = this.layers[i - 1]
            let output_layer = this.layers[i]

            result = this.solver.feed_forward(input_layer, output_layer);
        })

        return result
    }

//Network.prototype.train = function(data: &Vec<(Vec<f32>, Vec<f32>)>) -> f32 {
Network.prototype.train = function(data) {
    return Array(20000).fill(1)
        .map((_) => {
            return data.reduce((sum, [input, target]) => {
                return sum + this.train_pattern(input, target)
            }, 0.0) / data.length
        })
        .takeWhile((error) => error > 0.00075)
        .reduce((_, error) => error, 1.0)
}

Network.prototype.train_pattern = function(input, target) {
    this.run(input);
    this.calculate_deltas(target);
    this.adjust_weights();

    let output_layer = this.layers[this.layers.length - 1]

    this.mean_squared_error(output_layer.errors)
}

Network.prototype.calculate_deltas = function(target) {
    this.solver.target_output = target.slice()

    let layers_len = this.layers.length

    executeRange(0, layers_len, (i) => {
        let input_layer = this.layers[i + 1]
        let output_layer = this.layers[i]

        this.solver.calculate_deltas(input_layer, output_layer);
    })
}

Network.prototype.adjust_weights = function() {
    let layers_len = this.layers.length
    for (let i = 1; i < layers_len; i++) {
        let input_layer = this.layers[i - 1]
        let output_layer = this.layers[i]

        this.solver.adjust_weights(input_layer, output_layer)
    }
}

Network.prototype.mean_squared_error = function(errors) {
    return errors.reduce((sum, error) => sum + Math.pow(error, 2.0), 0.0) / errors.length
}


// main:

// A network which computes exclusive-or
let xor_net = new Network([2, 3, 1])

let data = [
    [[0.0, 0.0], [0.0]], // 0 ^ 0 == 0
    [[0.0, 1.0], [1.0]], // 0 ^ 1 == 1
    [[1.0, 1.0], [0.0]], // 1 ^ 1 == 0
    [[1.0, 0.0], [1.0]]  // 1 ^ 0 == 1
]

xor_net.train(data);

println("{:?}", Math.round(xor_net.run([0.0, 0.0]))); 
println("{:?}", Math.round(xor_net.run([0.0, 1.0]))); 
println("{:?}", Math.round(xor_net.run([1.0, 1.0])))
println("{:?}", Math.round(xor_net.run([1.0, 0.0])))
