#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <cassert>

using namespace std;

// 3D Tensor representation: [Channel][Height][Width]
typedef vector<vector<vector<double>>> Tensor3D;
// 1D Vector for fully connected layers
typedef vector<double> Vector1D;

// Helper to initialize a 3D Tensor with zeros
Tensor3D create_tensor(int c, int h, int w, double val = 0.0) {
    return Tensor3D(c, vector<vector<double>>(h, vector<double>(w, val)));
}

// Read integer from MNIST binary (Big Endian to Little Endian)
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Loads MNIST Images into a vector of 1-channel Tensors
vector<Tensor3D> load_mnist_images(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open file!");

    int magic_number=0, number_of_images=0, n_rows=0, n_cols=0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));

    number_of_images = reverseInt(number_of_images);
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);

    vector<Tensor3D> dataset(number_of_images, create_tensor(1, n_rows, n_cols));
    for (int i = 0; i < number_of_images; ++i) {
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                unsigned char temp=0;
                file.read((char*)&temp, sizeof(temp));
                dataset[i][0][r][c] = (double)temp / 255.0; // Normalize 0-1
            }
        }
    }
    return dataset;
}

// Loads MNIST labels
vector<int> load_mnist_labels(const string& filename) {
    ifstream file(filename, ios::binary);
    int magic_number=0, number_of_items=0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_items, sizeof(number_of_items));
    number_of_items = reverseInt(number_of_items);

    vector<int> labels(number_of_items);
    for (int i = 0; i < number_of_items; ++i) {
        unsigned char temp=0;
        file.read((char*)&temp, sizeof(temp));
        labels[i] = (int)temp;
    }
    return labels;
}

class ConvLayer {
public:
    int num_filters, filter_size, in_channels;
    vector<Tensor3D> filters; // [num_filters][in_channels][filter_size][filter_size]
    Vector1D biases;
    Tensor3D last_input;

    ConvLayer(int num_filters, int filter_size, int in_channels) 
        : num_filters(num_filters), filter_size(filter_size), in_channels(in_channels) {
        
        biases.resize(num_filters, 0.0);
        // He Initialization
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> d(0.0, sqrt(2.0 / (in_channels * filter_size * filter_size)));

        filters.resize(num_filters, create_tensor(in_channels, filter_size, filter_size));
        for(int f=0; f<num_filters; f++)
            for(int c=0; c<in_channels; c++)
                for(int i=0; i<filter_size; i++)
                    for(int j=0; j<filter_size; j++)
                        filters[f][c][i][j] = d(gen);
    }

    Tensor3D forward(const Tensor3D& input) {
        last_input = input;
        int in_h = input[0].size();
        int in_w = input[0][0].size();
        int out_h = in_h - filter_size + 1;
        int out_w = in_w - filter_size + 1;

        Tensor3D output = create_tensor(num_filters, out_h, out_w);

        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    double val = 0.0;
                    for (int c = 0; c < in_channels; c++) {
                        for (int fi = 0; fi < filter_size; fi++) {
                            for (int fj = 0; fj < filter_size; fj++) {
                                val += input[c][i+fi][j+fj] * filters[f][c][fi][fj];
                            }
                        }
                    }
                    output[f][i][j] = val + biases[f];
                }
            }
        }
        return output;
    }

    Tensor3D backward(const Tensor3D& d_L_d_out, double learning_rate) {
        int in_h = last_input[0].size();
        int in_w = last_input[0][0].size();
        int out_h = d_L_d_out[0].size();
        int out_w = d_L_d_out[0][0].size();

        // Gradients
        vector<Tensor3D> d_filters(num_filters, create_tensor(in_channels, filter_size, filter_size));
        Vector1D d_biases(num_filters, 0.0);
        Tensor3D d_L_d_input = create_tensor(in_channels, in_h, in_w);

        // Calculate gradients
        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    double grad = d_L_d_out[f][i][j];
                    d_biases[f] += grad;
                    
                    for (int c = 0; c < in_channels; c++) {
                        for (int fi = 0; fi < filter_size; fi++) {
                            for (int fj = 0; fj < filter_size; fj++) {
                                d_filters[f][c][fi][fj] += last_input[c][i+fi][j+fj] * grad;
                                d_L_d_input[c][i+fi][j+fj] += filters[f][c][fi][fj] * grad;
                            }
                        }
                    }
                }
            }
        }

        // Apply Gradient Descent updates
        for (int f = 0; f < num_filters; f++) {
            biases[f] -= learning_rate * d_biases[f];
            for (int c = 0; c < in_channels; c++) {
                for (int fi = 0; fi < filter_size; fi++) {
                    for (int fj = 0; fj < filter_size; fj++) {
                        filters[f][c][fi][fj] -= learning_rate * d_filters[f][c][fi][fj];
                    }
                }
            }
        }
        return d_L_d_input;
    }
};

class MaxPoolLayer {
public:
    int pool_size;
    Tensor3D last_input;

    MaxPoolLayer(int size) : pool_size(size) {}

    Tensor3D forward(const Tensor3D& input) {
        last_input = input;
        int c = input.size();
        int h = input[0].size() / pool_size;
        int w = input[0][0].size() / pool_size;
        
        Tensor3D output = create_tensor(c, h, w);

        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    double max_val = -1e9;
                    for (int pi = 0; pi < pool_size; pi++) {
                        for (int pj = 0; pj < pool_size; pj++) {
                            max_val = max(max_val, input[ch][i*pool_size + pi][j*pool_size + pj]);
                        }
                    }
                    output[ch][i][j] = max_val;
                }
            }
        }
        return output;
    }

    Tensor3D backward(const Tensor3D& d_L_d_out) {
        int c = last_input.size();
        int in_h = last_input[0].size();
        int in_w = last_input[0][0].size();
        
        Tensor3D d_L_d_input = create_tensor(c, in_h, in_w, 0.0);

        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < d_L_d_out[0].size(); i++) {
                for (int j = 0; j < d_L_d_out[0][0].size(); j++) {
                    
                    // Find the index of the max value in the forward pass to route the gradient
                    double max_val = -1e9;
                    int max_i = -1, max_j = -1;
                    for (int pi = 0; pi < pool_size; pi++) {
                        for (int pj = 0; pj < pool_size; pj++) {
                            double val = last_input[ch][i*pool_size + pi][j*pool_size + pj];
                            if (val > max_val) {
                                max_val = val;
                                max_i = i*pool_size + pi;
                                max_j = j*pool_size + pj;
                            }
                        }
                    }
                    // Pass gradient only to the max pixel
                    d_L_d_input[ch][max_i][max_j] = d_L_d_out[ch][i][j];
                }
            }
        }
        return d_L_d_input;
    }
};

class DenseLayer {
public:
    int in_features, out_features;
    vector<Vector1D> weights; // [out_features][in_features]
    Vector1D biases;
    Vector1D last_input;

    DenseLayer(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        weights.resize(out_features, Vector1D(in_features));
        biases.resize(out_features, 0.0);
        
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> d(0.0, sqrt(2.0 / in_features)); // He Init

        for(int i=0; i<out_features; i++)
            for(int j=0; j<in_features; j++)
                weights[i][j] = d(gen);
    }

    Vector1D forward(const Vector1D& input) {
        last_input = input;
        Vector1D output(out_features, 0.0);
        for(int i = 0; i < out_features; i++) {
            for(int j = 0; j < in_features; j++) {
                output[i] += weights[i][j] * input[j];
            }
            output[i] += biases[i];
        }
        return output;
    }

    Vector1D backward(const Vector1D& d_L_d_out, double learning_rate) {
        Vector1D d_L_d_input(in_features, 0.0);

        for(int i = 0; i < out_features; i++) {
            for(int j = 0; j < in_features; j++) {
                // Gradient w.r.t input
                d_L_d_input[j] += weights[i][j] * d_L_d_out[i];
                
                // Gradient w.r.t weights (and update immediately)
                double grad_w = d_L_d_out[i] * last_input[j];
                weights[i][j] -= learning_rate * grad_w;
            }
            // Update biases
            biases[i] -= learning_rate * d_L_d_out[i];
        }
        return d_L_d_input;
    }
};

// Helper to flatten 3D Tensor to 1D Vector
Vector1D flatten(const Tensor3D& t) {
    Vector1D v;
    for(const auto& c : t)
        for(const auto& row : c)
            for(double val : row)
                v.push_back(val);
    return v;
}

// Helper to reshape 1D Vector back to 3D Tensor
Tensor3D reshape(const Vector1D& v, int c, int h, int w) {
    Tensor3D t = create_tensor(c, h, w);
    int idx = 0;
    for(int ch=0; ch<c; ch++)
        for(int i=0; i<h; i++)
            for(int j=0; j<w; j++)
                t[ch][i][j] = v[idx++];
    return t;
}

int main() {
    cout << "Loading MNIST dataset..." << endl;
    
    // 1. Load the actual data
    vector<Tensor3D> train_images = load_mnist_images("train-images.idx3-ubyte");
    vector<int> train_labels = load_mnist_labels("train-labels.idx1-ubyte");
    
    cout << "Loaded " << train_images.size() << " images." << endl;

    // 2. Initialize Architecture
    ConvLayer conv(8, 3, 1);     
    MaxPoolLayer pool(2);        
    int flat_size = 8 * 13 * 13; 
    DenseLayer dense(flat_size, 10); 

    double learning_rate = 0.01;
    int epochs = 3;
    int batch_size = train_images.size(); // Or set to a smaller number like 1000 for testing

    cout << "Starting training..." << endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double loss_sum = 0;
        int correct = 0;

        // Loop over the actual dataset
        for (int i = 0; i < batch_size; i++) { 
            Tensor3D img = train_images[i];
            int label = train_labels[i];
            
            // --- FORWARD PASS ---
            Tensor3D conv_out = conv.forward(img);
            
            // ReLU
            for(auto& c : conv_out) 
                for(auto& row : c) 
                    for(auto& val : row) 
                        val = max(0.0, val);

            Tensor3D pool_out = pool.forward(conv_out);
            Vector1D flat_out = flatten(pool_out);
            Vector1D logits = dense.forward(flat_out);

            // Softmax
            double max_logit = *max_element(logits.begin(), logits.end());
            double sum_exp = 0.0;
            Vector1D probs(10);
            for(int j=0; j<10; j++) {
                probs[j] = exp(logits[j] - max_logit);
                sum_exp += probs[j];
            }
            for(int j=0; j<10; j++) probs[j] /= sum_exp;

            // Metrics
            loss_sum += -log(probs[label] + 1e-9);
            int pred = distance(probs.begin(), max_element(probs.begin(), probs.end()));
            if(pred == label) correct++;

            // --- BACKWARD PASS ---
            Vector1D d_L_d_logits = probs;
            d_L_d_logits[label] -= 1.0; 

            Vector1D d_L_d_flat = dense.backward(d_L_d_logits, learning_rate);
            Tensor3D d_L_d_pool = reshape(d_L_d_flat, 8, 13, 13);
            Tensor3D d_L_d_conv = pool.backward(d_L_d_pool);

            // ReLU derivative
            for(int ch=0; ch<8; ch++)
                for(int r=0; r<26; r++)
                    for(int c=0; c<26; c++)
                        if(conv_out[ch][r][c] <= 0) 
                            d_L_d_conv[ch][r][c] = 0.0;

            conv.backward(d_L_d_conv, learning_rate);
            
            // Optional: Print progress so you know it hasn't frozen
            if ((i + 1) % 1000 == 0) {
                cout << "Processed " << (i + 1) << " images..." << endl;
            }
        }
        cout << "Epoch " << epoch+1 << " - Avg Loss: " << loss_sum/batch_size << " | Acc: " << (double)correct/batch_size * 100.0 << "%" << endl;
    }
    return 0;
}