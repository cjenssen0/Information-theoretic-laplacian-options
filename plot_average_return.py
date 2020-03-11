import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('file', type=str,
                    help='Name of average return file to load.')
parser.add_argument('--xlim', type=int, default=500,
                    help='Name of average return file to load.')

args = parser.parse_args()

if __name__ == "__main__":
   # results = np.load('data_files/average_return.npy')
   # results = np.load('data_files/average_return_K.npy')
   results = np.load(args.file)

   plt.show()
   x_legend = range(len(results[0][:]))
   graph_agent_0, = plt.plot(x_legend, results[0][:], label="primitive actions")
   graph_agent_2, = plt.plot(x_legend, results[2][:], label="2 option")
   graph_agent_4, = plt.plot(x_legend, results[4][:], label="4 options")
   graph_agent_8, = plt.plot(x_legend, results[8][:], label="8 options")
   graph_agent_64, = plt.plot(x_legend, results[30][:], label="64 options")
#    graph_agent_128, = plt.plot(x_legend, results[128][:], label="128 options")
#    graph_agent_200, = plt.plot(x_legend, results[200][:], label="200 options")

   plt.legend(handles=[graph_agent_0, graph_agent_2, graph_agent_4, graph_agent_8, graph_agent_64])#, graph_agent_128, graph_agent_200])
   plt.xlabel('Episodes')
   plt.ylabel('Average return')
   plt.xlim(0, args.xlim)
   plt.show()
