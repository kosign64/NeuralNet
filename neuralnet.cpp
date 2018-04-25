#include "neuralnet.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <ctime>
#include <clocale>
#include <algorithm>

#ifdef WITH_TBB
#include <tbb/tbb.h>

using namespace tbb;
#endif

NeuralNet::NeuralNet(double eta, double alpha) : recentAverageError_(0),
    recentAverageSmoothingFactor_(0.5)
{
    Neuron::setEta(eta);
    Neuron::setAlpha(alpha);
}

NeuralNet::NeuralNet(const vector<size_t> &topology, double eta, double alpha) : recentAverageError_(0),
    recentAverageSmoothingFactor_(0.5)
{
    srand(time(NULL));
    createNet(topology);
    Neuron::setEta(eta);
    Neuron::setAlpha(alpha);
}

//=============================================================================

void NeuralNet::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == (layers_[0].size() - 1));
    const size_t size = inputVals.size();

#ifdef WITH_TBB
    parallel_for(size_t(0), size, [&](size_t i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    });
#else
    for(size_t i = 0; i < size; ++i)
    {
        layers_[0][i].setOutputVal(inputVals[i]);
    }
#endif
    for(size_t layerNum = 1; layerNum < layers_.size(); ++layerNum)
    {
        Layer &prevLayer = layers_[layerNum - 1];
        const size_t layerSize = layers_[layerNum].size() - 1;

#ifdef WITH_TBB
        parallel_for(size_t(0), layerSize, [&](size_t neuronNum){
                         m_layers[layerNum][neuronNum].feedForward(prevLayer);
                     });
#else
        for(size_t neuronNum = 0; neuronNum < layerSize; ++neuronNum)
        {
            layers_[layerNum][neuronNum].feedForward(prevLayer);
        }
#endif
    }
}

void NeuralNet::backProp(const vector<double> &targetVals, bool update)
{
    assert(targetVals.size() == (layers_.back().size() - 1));

    Layer &outputLayer = layers_.back();
    error_ = 0.0;

    for(size_t n = 0; n < (outputLayer.size() - 1); ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        error_ += delta * delta;
    }
    error_ /= (outputLayer.size() - 1);
    error_ = sqrt(error_);

    recentAverageError_ =
            (recentAverageError_ * recentAverageSmoothingFactor_ + error_) /
            (recentAverageSmoothingFactor_ + 1.0);

    // Output Layer Gradients
    const size_t outputLayerSize = outputLayer.size() - 1;
#ifdef WITH_TBB
    parallel_for(size_t(0), outputLayerSize, [&](size_t n){
        outputLayer[n].calcOutputGradient(targetVals[n]);
    });
#else
    for(size_t n = 0; n < outputLayerSize; ++n)
    {
        outputLayer[n].calcOutputGradient(targetVals[n]);
    }
#endif

    // Hidden Layers Gradients
    for(size_t layerNum = layers_.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = layers_[layerNum];
        Layer &nextLayer = layers_[layerNum + 1];

        const size_t hiddenLayerSize = hiddenLayer.size();
#ifdef WITH_TBB
        parallel_for(size_t(0), hiddenLayerSize, [&](size_t n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        });
#else
        for(size_t n = 0; n < hiddenLayerSize; ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
#endif
    }

    // Update connection weights
    for(size_t layerNum = layers_.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = layers_[layerNum];
        Layer &prevLayer = layers_[layerNum - 1];

        const size_t layerSize = layer.size() - 1;
#ifdef WITH_TBB
        parallel_for(size_t(0), layerSize, [&](size_t n){
            layer[n].calcInputWeightsUpdates(prevLayer, update);
        });
#else
        for(size_t n = 0; n < layerSize; ++n)
        {
            layer[n].calcInputWeightsUpdates(prevLayer, update);
        }
#endif
    }
}

void NeuralNet::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for(size_t n = 0; n < layers_.back().size() - 1; ++n)
    {
        resultVals.push_back(layers_.back()[n].getOutputVal());
    }
}

double NeuralNet::train(size_t iterations, double error,
                        vector<array<vector<double>, 2>> &trainData, size_t batchSize)
{
    const size_t size = trainData.size();
    double averageError = 0.0;
    for(size_t i = 0; (i < iterations); ++i)
    {
        random_shuffle(trainData.begin(), trainData.end());
        averageError = 0;
        for(size_t n = 0; n < size; ++n)
        {
            bool update = false;
            if((n != 0) && (((n % batchSize) == 0) || (n == size - 1)))
            {
                update = true;
            }
            feedForward(trainData[n][0]);
            backProp(trainData[n][1], update);
            averageError += error_;
        }
        if((i % 30) == 0)
        {
            cout << "Iteration: " << i << " Average Error sum: " << averageError <<
                    " Average Error: " << (averageError / size) << endl;
        }
        if((averageError / size) < error) break;
    }

    return averageError / size;
}

//=============================================================================

void NeuralNet::createNet(const vector<size_t> &topology)
{
    size_t numberOfLayers = topology.size();
    for(size_t layerNum = 0; layerNum < numberOfLayers; ++layerNum)
    {
        layers_.push_back(Layer());
        for(size_t neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            size_t numOutputs = (layerNum == (numberOfLayers - 1)) ? 0 : topology[layerNum + 1];
            layers_.back().push_back(Neuron(numOutputs, neuronNum));
        }

        layers_.back().back().setOutputVal(1.0);
    }
}

void NeuralNet::saveNet(const char *filename) const
{
    setlocale(LC_ALL, "C");
    ofstream file;
    file.open(filename, ios::out);
    if(file.is_open())
    {
        for(size_t l = 0; l < layers_.size(); ++l)
        {
            file << layers_[l].size() - 1 << ";";
        }
        file << endl;
        for(size_t l = 0; l < layers_.size() - 1; ++l)
        {
            const Layer &layer = layers_[l];
            for(size_t n = 0; n < layer.size(); ++n)
            {
                vector <Connection> weights = layer[n].getWeights();
                for(size_t i = 0; i < weights.size(); ++i)
                {
                    file << weights[i].weight << ";";
                }
            }
            file << endl;
        }
        file.close();
    }
    else
    {
        cerr << "Can't open file for save net" << endl << flush;
        exit(-1);
    }
}
void NeuralNet::loadNet(const char *filename)
{
    setlocale(LC_ALL, "C");
    ifstream file;
    file.open(filename, ios::in);
    if(file.is_open())
    {
        if(layers_.size() != 0)
        {
            layers_.clear();
        }
        string line;
        getline(file, line);
        vector<size_t> topology;
        {
            istringstream stream(line);
            while(stream)
            {
                string str;
                getline(stream, str, ';');
                topology.push_back(stoi(str));
            }
        }
        topology.pop_back();
        createNet(topology);

        for(size_t layerNumber = 0; layerNumber < (layers_.size() - 1); ++layerNumber)
        {
            Layer &layer = layers_[layerNumber];
            Layer &nextLayer = layers_[layerNumber + 1];
            vector <double> layerWeights;
            if(!getline(file, line))
            {
                cerr << "Error while loading weights for layer" << layerNumber <<
                        " of" << layers_.size() << endl << flush;
                exit(-1);
            }
            istringstream stream(line);
            while(stream)
            {
                string str;
                getline(stream, str, ';');
                layerWeights.push_back(atof(str.c_str()));
            }
            layerWeights.pop_back();
            size_t j = 0;
            for(size_t neuronNumber = 0; neuronNumber < layer.size(); ++neuronNumber)
            {
                vector <double> weights;
                Neuron &neuron = layer[neuronNumber];
                for(size_t i = 0; i < nextLayer.size() - 1; ++i)
                {
                    weights.push_back(layerWeights[j++]);
                }
                neuron.setWeights(weights);
            }
        }
    }
    else
    {
        cerr << "Can't open file for load net" << endl << flush;
        exit(-1);
    }

    file.close();
}
