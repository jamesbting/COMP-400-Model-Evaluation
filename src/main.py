from champion_net import ChampionNet
import torch
import numpy as np
import csv
model = "../nn-reward-function/models/champion-model-06-04-2021-1617748751-ls128-lr0.0005-l20.05/model.pickle"
results_files = ['../mcts/results/recs-nn-06-04-2021-1617755432.csv']
#results_files = ['../mcts/results/recs-random-06-04-2021-1617755339.csv', '../mcts/results/recs-random-06-04-2021-1617736035.csv']
def main():
    net = load_nn(model, 128)
    recs = []
    for file in results_files:
        load_recommendations(recs, file)
        
    mean_win_rate = calculate_mean_win_rate(net, recs)
    print(mean_win_rate)

def load_recommendations(res, filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter =',')
        for line in reader:
            res.append(torch.Tensor(convert_to_state(line)))
        
def convert_to_state(combination):
    res = [0 for i in range(154)] #154 champions
    for i in range(5):
        res[int(combination[i])] = 1
    
    for i in range(5, 10):
        res[int(combination[i])] = -1
    return res

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

if __name__ == '__main__':
    main()