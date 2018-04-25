#include "neuralnet.h"
#include <vector>
#include <array>
#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    vector <size_t> topology;
    topology.push_back(1);
    topology.push_back(5);
    topology.push_back(1);
    NeuralNet net(topology, 0.6);

    vector< vector<double> > inputs;
    vector< vector<double> > outputs;

    for(int angle = 0; angle < 360; angle += 5)
    {
        vector<double> angleIn;
        vector<double> sinOut;
        angleIn.push_back((double)angle / 360.);
        sinOut.push_back(sin((double)angle * M_PI / 180.));
        inputs.push_back(angleIn);
        outputs.push_back(sinOut);
    }

    vector<array<vector<double>, 2>> data;

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        array<vector<double>, 2> tmp{inputs[i], outputs[i]};
        data.push_back(move(tmp));
    }

    double error = net.train(100000, 0.018, data, 8);

    cout << "Error: " << error << endl;

    double angle = 39.6;
    vector <double> input;
    vector <double> output;
    input.push_back(angle / 360.);
    cout << "Input: " << input[0] << endl;
    output.resize(1);

    net.feedForward(input);
    net.getResults(output);

    cout << sin(angle * M_PI / 180.) << " " << output[0] << endl;

    net.saveNet("net.net");

    return 0;
}
