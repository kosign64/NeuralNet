#include "neuron.h"
#include <cmath>
#include <cassert>
#include <iostream>

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(size_t numOutputs, size_t index) : index_(index)
{
    for(size_t c = 0; c < numOutputs; ++c)
    {
        outputWeights_.push_back(Connection());
        outputWeights_.back().weight = randomWeight() / 20;
    }
}

//=============================================================================

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    const size_t size = prevLayer.size();
    for(size_t n = 0; n < size; ++n)
    {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights_[index_].weight;
    }

    outputVal_ = transferFunction(sum);
}

double Neuron::transferFunction(double sum)
{
    return tanh(sum);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x * x;
}

void Neuron::calcOutputGradient(double targetVal)
{
    double delta = targetVal - outputVal_;
    gradient_ = delta * transferFunctionDerivative(outputVal_);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDow(nextLayer);
    gradient_ = dow * transferFunctionDerivative(outputVal_);
}

double Neuron::sumDow(const Layer &nextLayer) const
{
    double sum = 0.0;

    for(size_t n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += outputWeights_[n].weight * nextLayer[n].gradient_;
    }

    return sum;
}

void Neuron::calcInputWeightsUpdates(Layer &prevLayer, bool update)
{
    for(size_t n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        vector<double> &corrections = neuron.outputWeights_[index_].corrections;
        double oldDeltaWeight = neuron.outputWeights_[index_].deltaWeight;

        double newDeltaWeight = eta * neuron.outputVal_ * gradient_ +
                alpha * oldDeltaWeight;
        corrections.push_back(newDeltaWeight);
        if(update)
        {
            double sum = 0;
            for(const auto &c : corrections)
            {
                sum += c;
            }
            newDeltaWeight = sum / corrections.size();
            neuron.outputWeights_[index_].deltaWeight = newDeltaWeight;
            neuron.outputWeights_[index_].weight += newDeltaWeight;
            corrections.clear();
        }
    }
}

//=============================================================================

void Neuron::setWeights(const vector <double> &weights)
{
    assert(weights.size() == outputWeights_.size());

    for(size_t i = 0; i < outputWeights_.size(); ++i)
    {
        outputWeights_[i].weight = weights[i];
    }
}
