#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include "neuron.h"

// May increase performance for large networks
// For small networks may dramatically decrease performance
//#define WITH_TBB

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

class NeuralNet
{
public:
    NeuralNet(double eta = 0.05, double alpha = 0.1);
    NeuralNet(const vector<size_t> &topology,
              double eta = 0.05, double alpha = 0.1);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double train(size_t iterations, double error, const vector< vector<double> > &inputs,
                 const vector< vector<double> > &outputs);

    void saveNet(const char *filename) const;
    void loadNet(const char *filename);

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
    void createNet(const vector<size_t> &topology);
};

#endif // NEURALNET_H
