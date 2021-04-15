from champion_net import ChampionNet
import torch
import numpy as np
import csv
import math

config = {
    "model":  "../nn-reward-function/models/champion-model-06-04-2021-1617757457-ls256-lr0.0005-l20.05/model.pickle",
    "recs_files": ['../mcts/results/recs-nn-06-04-2021-1617771805.csv'],
    "results_files":  ['../mcts/results/results-random-06-04-2021-1617771187.csv'],
    'percentile': 0.99
}



def main():
    net = load_nn(config["model"], 256)
    recs = []
    for file in config["recs_files"]:
        load_recommendations(recs, file)
    latencies = []
    memory = []
    for file in config["results_files"]:
        load_data(latencies, memory, file)

    mean_win_rate = calculate_mean_win_rate(net, recs)
    print(f'Mean win rate: {mean_win_rate}')
    k = config['percentile']
    latency_percentile = percentile(latencies, k)
    print(f'Latency {k*100}-th percentile: {latency_percentile}')

#load the recommendations from each file and append it to res
def load_recommendations(res, filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for line in reader:
            res.append(torch.Tensor(convert_to_state(line)))
        
def load_data(latencies, memory, filename):
     with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for line in reader:
            latencies.append(float(line[0]))
            memory.append(float(line[1]))

#convert to long state for the model
def convert_to_state(combination):
    res = [0 for i in range(154)] #154 champions
    for i in range(5):
        res[int(combination[i])] = 1
    
    for i in range(5, 10):
        res[int(combination[i])] = -1
    return res

#compute the mean win rate all recs
def calculate_mean_win_rate(net, recs):
    sum = 0
    for rec in recs:
        sum += net(rec).tolist()[0]
    return sum / len(recs)

#nn: get the neural network
def load_nn(filename, num_units):
    net_dict = torch.load(filename)
    model = ChampionNet(num_units)
    model.load_state_dict(net_dict)
    model.eval()
    return model

def percentile(data, k):
    if k < 0 or k > 1:
        raise ValueError
    data.sort()
    i = math.floor(len(data) * k)
    return data[i]

if __name__ == '__main__':
    main()