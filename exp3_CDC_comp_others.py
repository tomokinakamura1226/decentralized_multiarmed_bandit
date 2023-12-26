import numpy as np
import random
import math
import copy
import networkx as nx
import matplotlib.pyplot as plt
from progressbar import ProgressBar

# Seed for random variables
np.random.seed(1)

class Bandit():
    def __init__(self, playtimes, gamma, eta, K, player_number):
        self.playtimes = playtimes
        self.arms = K
        self.gamma = gamma
        self.eta = eta
        self.weight = K ** (-(1 / player_number))
        self.count = 0
        #self.beta = beta
        self.player_number = player_number
        self.k = int(np.ceil(self.player_number/2))
        self.alpha = 1

    def play(self):
        nu = 1/self.player_number
        player_list = [1 for i in range(self.player_number)]
        X = [[0 for i in range(self.arms)] for j in range(self.player_number)]
        X_hat = [[0 for i in range(self.arms)] for j in range(len(player_list))]
        # a = [[0.1 for i in range(len(player_list))] for j in range(len(player_list))]
        w = [[1 / (self.arms ** nu) for i in range(self.arms)] for j in range(len(player_list))]
        W = [self.arms**(1-nu) for i in range(len(player_list))]
        #print(w)
        #print(W)
        #print(np.sum(w[0])/W[0])
        p = [[(1 / self.arms) for i in range(self.arms)] for j in range(len(player_list))]
        selected_arm_X = [[0 for i in range(self.playtimes)] for j in range(len(player_list))]
        selected_arm_k = [[0 for i in range(self.playtimes)] for j in range(len(player_list))]
        G = [[0 for i in range(self.playtimes)] for j in range(len((player_list)))]
        G_star = [0 for i in range(self.playtimes)]
        G_accumulation = [0 for i in range(self.playtimes)]
        G_star_accumulation = [0 for i in range(self.playtimes)]
        aX = [[0 for i in range(self.arms)] for j in range(len(player_list))]
        Regret = [0 for i in range(self.playtimes)]
        Regret_ = [0 for i in range(self.playtimes)]
        X_table = [[0 for i in range(self.arms)] for j in range(len(player_list))]

        prog = ProgressBar(max_value=self.playtimes)
        for t in range(self.playtimes):
            prog.update(t)
            # Open MAS
            # if t % 100 == 0 and t > 0:
            #     for i in range(0, int(self.player_number / 2)):
            #         player_list[i] = random.randint(0, 1)

            Graph = nx.random_k_out_graph(self.player_number, self.k, self.alpha, self_loops=False, seed=None)
            Adj = nx.to_numpy_array(Graph)
            Adj = Adj + np.eye(self.player_number)
            # print(Adj)
            max_degree = 0
            for i in range(self.player_number):
                if (Graph.in_degree[i] > max_degree):
                    max_degree = Graph.in_degree[i]

            atmp = np.zeros(self.player_number)
            for i in range(self.player_number):
                atmp[i] = copy.copy(1.0 / max_degree)

                # a = np.zeros(self.n)
                # for i in range(self.n):
                #     a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))
            a = np.zeros((self.player_number, self.player_number))  # 確率行列(重み付き)
            for i in range(self.player_number):
                for j in range(self.player_number):
                    if i != j and Adj[i][j] > 0:
                        a[i][j] = copy.copy(atmp[i])
                        #a_ij = min(atmp[i], atmp[j])
                        #a[i][j] = copy.copy(a_ij)
                        #a[j][i] = copy.copy(a_ij)

            for i in range(self.player_number):
                sum = 0.0
                for j in range(self.player_number):
                    sum += a[i][j]
                a[i][i] = 1.0 - sum

            # print(a)

            if t % 1000 == 0:
                plt.figure()
                pos = nx.spring_layout(Graph)
                labels = {}
                for i in range(self.player_number):
                    labels[i] = r"{0}".format(i + 1)
                #nx.draw_networkx_nodes(Graph, pos, node_size=120, alpha=1.0, node_color="blue")
                nx.draw_networkx_nodes(Graph, pos, node_size=400, alpha=1.0, node_color="lightblue")
                nx.draw_networkx_edges(Graph, pos, width=1, arrowstyle='->', arrowsize=20)
                nx.draw_networkx_labels(Graph, pos, labels, font_size=18)
                plt.axis('off')
                plt.savefig("network_{0}".format(self.player_number)+"_{0}".format(t)+".png")
                # plt.savefig("network{0}.eps".format(t))
                # plt.show()

            count = 0
            for i in range(self.player_number):
                for j in range(len(X[i])):
                    if j == 0:
                        X[i][j] = random.uniform(0.8, 1.0)
                    elif i % 2 == 0:
                        if j % 2 == 0:
                            X[i][j] = random.uniform(0.1, 0.6)
                        else:
                            X[i][j] = random.uniform(0.4, 1.0)
                    else:
                        if j % 2 == 0:
                            X[i][j] = random.uniform(0.4, 1.0)
                        else:
                            X[i][j] = random.uniform(0.1, 0.6)


            for i in range(len(player_list)):
                if player_list[i] == 0:
                    pass
                else:
                    count = count + 1

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    for k in range(self.arms):
                        p[i][k] = (1 - self.gamma) * w[i][k] / W[i] + self.gamma / self.arms
                        #p[i][k] = (1 - (self.gamma / self.player_number)) * w[i][k] + self.gamma / (self.arms * self.player_number)
                else:
                    pass


            for i in range(len(player_list)):
                if player_list[i] == 1:
                    selected_arm_X[i][t] = np.random.choice(a=X[i], size=1, p=p[i])
                    for j in range(self.arms):
                        if selected_arm_X[i][t] == X[i][j]:
                            selected_arm_k[i][t] = j

            for i in range(len(player_list)):
                for k in range(self.arms):
                    if player_list[i] == 1:
                        if k == selected_arm_k[i][t]:
                            X_table[i][k] = X[i][k]
                        else:
                            X_table[i][k] = 0

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    for k in range(self.arms):
                        if k == selected_arm_k[i][t]:
                            for j in range(len(player_list)):
                                if selected_arm_k[j][t] == selected_arm_k[i][t]:
                                    aX[i][k] = aX[i][k] + a[i][j] * X_table[j][k]
                        else:
                            aX[i][k] = 0
                        X_hat[i][k] = aX[i][k] / p[i][k]
                        aX[i][k] = 0

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    for k in range(self.arms):
                        w[i][k] = w[i][k] * math.exp(self.eta * X_hat[i][k])

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    W[i] = 0
                    for k in range(self.arms):
                        W[i] = W[i] + w[i][k]

            # for i in range(len(player_list)):
            #     if player_list[i] == 1:
            #         for k in range(self.arms):
            #             w[i][k] = w[i][k] / W[i]

            G_sum = [0 for i in range(self.playtimes)]
            for i in range(len(player_list)):
                for j in range(len(player_list)):
                    for k in range(self.arms):
                        if player_list[i] == 1:
                            if player_list[j] == 1:
                                G[i][t] = p[j][k] * a[i][j] * X[j][k] + G[i][t]
                G_sum[t] = G_sum[t] + G[i][t]


            if t > 0:
                G_accumulation[t] = G_sum[t] + G_accumulation[t - 1]
                G_star_accumulation[t] = G_star_accumulation[t - 1]
                for i in range(len(player_list)):
                    if player_list[i] == 1:
                        G_star_accumulation[t] = G_star_accumulation[t] + 0.9
                # print(G_accumulation[t])

        for t in range(self.playtimes):
            Regret[t] = G_star_accumulation[t] - G_accumulation[t]
            '''print("G_star[t],G_acuumulation[t]",G_star[t] , G_acuumulation[t],t)'''

        return Regret

    def play_exp3coop(self):
        nu = 1/self.player_number
        player_list = [1 for i in range(self.player_number)]
        X = [[0 for i in range(self.arms)] for j in range(self.player_number)]
        X_hat = [[0 for i in range(self.arms)] for j in range(len(player_list))]
        # a = [[0.1 for i in range(len(player_list))] for j in range(len(player_list))]
        w = [[1 / (self.arms ** nu) for i in range(self.arms)] for j in range(len(player_list))]
        W = [self.arms**(1-nu) for i in range(len(player_list))]
        #print(w)
        #print(W)
        #print(np.sum(w[0])/W[0])
        p = [[(1 / self.arms) for i in range(self.arms)] for j in range(len(player_list))]
        q = [[(1 / self.arms) for i in range(self.arms)] for j in range(len(player_list))]
        selected_arm_X = [[0 for i in range(self.playtimes)] for j in range(len(player_list))]
        selected_arm_k = [[0 for i in range(self.playtimes)] for j in range(len(player_list))]
        G = [[0 for i in range(self.playtimes)] for j in range(len((player_list)))]
        G_star = [0 for i in range(self.playtimes)]
        G_accumulation = [0 for i in range(self.playtimes)]
        G_star_accumulation = [0 for i in range(self.playtimes)]
        aX = [[0 for i in range(self.arms)] for j in range(len(player_list))]
        Regret = [0 for i in range(self.playtimes)]
        Regret_ = [0 for i in range(self.playtimes)]
        X_table = [[0 for i in range(self.arms)] for j in range(len(player_list))]

        prog = ProgressBar(max_value=self.playtimes)
        for t in range(self.playtimes):
            prog.update(t)

            Graph = nx.random_k_out_graph(self.player_number, self.k, self.alpha, self_loops=False, seed=None)
            Adj = nx.to_numpy_array(Graph)
            Adj = Adj + np.eye(self.player_number)
            # print(Adj)
            max_degree = 0
            for i in range(self.player_number):
                if (Graph.in_degree[i] > max_degree):
                    max_degree = Graph.in_degree[i]

            atmp = np.zeros(self.player_number)
            for i in range(self.player_number):
                atmp[i] = copy.copy(1.0 / max_degree)

                # a = np.zeros(self.n)
                # for i in range(self.n):
                #     a[i] = copy.copy(1.0 / (nx.degree(self.G)[i] + 1.0))
            a = np.zeros((self.player_number, self.player_number))  # 確率行列(重み付き)
            for i in range(self.player_number):
                for j in range(self.player_number):
                    if i != j and Adj[i][j] > 0:
                        a[i][j] = copy.copy(atmp[i])
                        #a_ij = min(atmp[i], atmp[j])
                        #a[i][j] = copy.copy(a_ij)
                        #a[j][i] = copy.copy(a_ij)

            for i in range(self.player_number):
                sum = 0.0
                for j in range(self.player_number):
                    sum += a[i][j]
                a[i][i] = 1.0 - sum

            # print(a)

            if t % 1000 == 0:
                plt.figure()
                pos = nx.spring_layout(Graph)
                labels = {}
                for i in range(self.player_number):
                    labels[i] = r"{0}".format(i + 1)
                #nx.draw_networkx_nodes(Graph, pos, node_size=120, alpha=1.0, node_color="blue")
                nx.draw_networkx_nodes(Graph, pos, node_size=400, alpha=1.0, node_color="lightblue")
                nx.draw_networkx_edges(Graph, pos, width=1, arrowstyle='->', arrowsize=20)
                nx.draw_networkx_labels(Graph, pos, labels, font_size=18)
                plt.axis('off')
                plt.savefig("network_{0}".format(self.player_number)+"_{0}".format(t)+".png")
                # plt.savefig("network{0}.eps".format(t))
                # plt.show()

            count = 0
            for i in range(self.player_number):
                for j in range(len(X[i])):
                    if j == 0:
                        X[i][j] = random.uniform(0.8, 1.0)
                    elif i % 2 == 0:
                        if j % 2 == 0:
                            X[i][j] = random.uniform(0.1, 0.6)
                        else:
                            X[i][j] = random.uniform(0.4, 1.0)
                    else:
                        if j % 2 == 0:
                            X[i][j] = random.uniform(0.4, 1.0)
                        else:
                            X[i][j] = random.uniform(0.1, 0.6)


            for i in range(len(player_list)):
                if player_list[i] == 0:
                    pass
                else:
                    count = count + 1

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    selected_arm_X[i][t] = np.random.choice(a=X[i], size=1, p=p[i])
                    for j in range(self.arms):
                        if selected_arm_X[i][t] == X[i][j]:
                            selected_arm_k[i][t] = j

            for i in range(len(player_list)):
                for k in range(self.arms):
                    if player_list[i] == 1:
                        if k == selected_arm_k[i][t]:
                            X_table[i][k] = X[i][k]
                        else:
                            X_table[i][k] = 0

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    W[i] = 0
                    for k in range(self.arms):
                        W[i] = W[i] + w[i][k]

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    for k in range(self.arms):
                        p[i][k] = w[i][k] / W[i]
                        # p[i][k] = w[i][k] / W[i] + self.gamma / self.arms
                        #p[i][k] = (1 - (self.gamma / self.player_number)) * w[i][k] + self.gamma / (self.arms * self.player_number)
                else:
                    pass

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    for k in range(self.arms):
                        w[i][k] = p[i][k] * math.exp(self.eta * X_hat[i][k])

            for i in range(len(player_list)):
                if player_list[i] == 1:
                    for k in range(self.arms):
                        X_hat[i][k] = X_table[i][k] / q[i][k]

            for i in range(self.player_number):
                for k in range(self.arms):
                    qtmp = 1
                    for j in range(self.player_number):
                        if Adj[i][j] > 0:
                            qtmp = qtmp * (1 - p[j][k])
                        else:
                            qtmp = 0
                    q[i][k] = 1 - qtmp

            # for i in range(len(player_list)):
            #     if player_list[i] == 1:
            #         for k in range(self.arms):
            #             if k == selected_arm_k[i][t]:
            #                 for j in range(len(player_list)):
            #                     if selected_arm_k[j][t] == selected_arm_k[i][t]:
            #                         aX[i][k] = aX[i][k] + a[i][j] * X_table[j][k]
            #             else:
            #                 aX[i][k] = 0
            #             X_hat[i][k] = aX[i][k] / p[i][k]
            #             aX[i][k] = 0
            #
            # for i in range(len(player_list)):
            #     if player_list[i] == 1:
            #         for k in range(self.arms):
            #             w[i][k] = w[i][k] * math.exp(self.eta * X_hat[i][k])
            #


            G_sum = [0 for i in range(self.playtimes)]
            for i in range(len(player_list)):
                for j in range(len(player_list)):
                    for k in range(self.arms):
                        if player_list[i] == 1:
                            if player_list[j] == 1:
                                G[i][t] = p[j][k] * a[i][j] * X[j][k] + G[i][t]
                G_sum[t] = G_sum[t] + G[i][t]


            if t > 0:
                G_accumulation[t] = G_sum[t] + G_accumulation[t - 1]
                G_star_accumulation[t] = G_star_accumulation[t - 1]
                for i in range(len(player_list)):
                    if player_list[i] == 1:
                        G_star_accumulation[t] = G_star_accumulation[t] + 0.9
                # print(G_accumulation[t])

        for t in range(self.playtimes):
            Regret[t] = G_star_accumulation[t] - G_accumulation[t]
            '''print("G_star[t],G_acuumulation[t]",G_star[t] , G_acuumulation[t],t)'''

        return Regret


if __name__ == '__main__':
    # playtime = 20000
    playtime = 20000
    repeat_time = 1
    learning_pram = [0.005, 0.005, 0.01, 0.01, 0.02, 0.02]
    label_legend = ['proposed', 'Exp3-Coop', 'proposed', 'Exp3-Coop', 'proposed', 'Exp3-Coop']
    game1_Regret_t = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game1_Regret_ave = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game2_Regret_t = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game2_Regret_ave = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game3_Regret_t = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game3_Regret_ave = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game4_Regret_t = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game4_Regret_ave = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game5_Regret_t = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game5_Regret_ave = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game6_Regret_t = [[0 for i in range(repeat_time)] for j in range(playtime)]
    game6_Regret_ave = [[0 for i in range(repeat_time)] for j in range(playtime)]
    for t_1 in range(repeat_time):
        '''game1 = Bandit(playtime, 0.01, 0.04, 10)'''
        '''Bandit(playtimes, gamma, eta, K, player_number)'''
        # game1 = Bandit(playtime, 0.1, 0.0005, 100, 10)
        game1 = Bandit(playtime, 0.01, learning_pram[0], 10, 10)
        game1_Regret = game1.play()
        for t_2 in range(playtime):
            game1_Regret_t[t_2][t_1] = game1_Regret[t_2]
    for t in range(repeat_time):
        for i in range(playtime):
            game1_Regret_ave[i] = sum(game1_Regret_t[i]) / len(game1_Regret_t[i])

    for t_1 in range(repeat_time):
        '''game1 = Bandit(playtime, 0.01, 0.04, 10)'''
        '''Bandit(playtimes, gamma, eta, K, player_number)'''
        # game2 = Bandit(playtime, 0.2, 0.0005, 100, 20)
        game2 = Bandit(playtime, 0.01, learning_pram[1], 10, 10)
        # game2_Regret = game2.play()
        game2_Regret = game2.play_exp3coop()
        for t_2 in range(playtime):
            game2_Regret_t[t_2][t_1] = game2_Regret[t_2]
    for t in range(repeat_time):
        for i in range(playtime):
            game2_Regret_ave[i] = sum(game2_Regret_t[i]) / len(game2_Regret_t[i])

    for t_1 in range(repeat_time):
        '''game1 = Bandit(playtime, 0.01, 0.04, 10)'''
        '''Bandit(playtimes, gamma, eta, K, player_number)'''
        # game3 = Bandit(playtime, 0.3, 0.0005, 100, 30)
        game3 = Bandit(playtime, 0.01, learning_pram[2], 10, 10)
        game3_Regret = game3.play()
        for t_2 in range(playtime):
            game3_Regret_t[t_2][t_1] = game3_Regret[t_2]
    for t in range(repeat_time):
        for i in range(playtime):
            game3_Regret_ave[i] = sum(game3_Regret_t[i]) / len(game3_Regret_t[i])

    for t_1 in range(repeat_time):
        '''game1 = Bandit(playtime, 0.01, 0.04, 10)'''
        '''Bandit(playtimes, gamma, eta, K, player_number)'''
        # game4 = Bandit(playtime, 0.4, 0.0005, 100, 40)
        game4 = Bandit(playtime, 0.01, learning_pram[3], 10, 10)
        # game4_Regret = game4.play()
        game4_Regret = game4.play_exp3coop()
        for t_2 in range(playtime):
            game4_Regret_t[t_2][t_1] = game4_Regret[t_2]
    for t in range(repeat_time):
        for i in range(playtime):
            game4_Regret_ave[i] = sum(game4_Regret_t[i]) / len(game4_Regret_t[i])

    for t_1 in range(repeat_time):
        '''game1 = Bandit(playtime, 0.01, 0.04, 10)'''
        '''Bandit(playtimes, gamma, eta, K, player_number)'''
        # game3 = Bandit(playtime, 0.3, 0.0005, 100, 30)
        game5 = Bandit(playtime, 0.01, learning_pram[4], 10, 10)
        game5_Regret = game5.play()
        for t_2 in range(playtime):
            game5_Regret_t[t_2][t_1] = game5_Regret[t_2]
    for t in range(repeat_time):
        for i in range(playtime):
            game5_Regret_ave[i] = sum(game5_Regret_t[i]) / len(game5_Regret_t[i])

    for t_1 in range(repeat_time):
        '''game1 = Bandit(playtime, 0.01, 0.04, 10)'''
        '''Bandit(playtimes, gamma, eta, K, player_number)'''
        # game4 = Bandit(playtime, 0.4, 0.0005, 100, 40)
        game6 = Bandit(playtime, 0.01, learning_pram[5], 10, 10)
        # game4_Regret = game4.play()
        game6_Regret = game6.play_exp3coop()
        for t_2 in range(playtime):
            game6_Regret_t[t_2][t_1] = game6_Regret[t_2]
    for t in range(repeat_time):
        for i in range(playtime):
            game6_Regret_ave[i] = sum(game6_Regret_t[i]) / len(game6_Regret_t[i])

    fig = plt.figure()
    plt.plot(range(playtime), game1_Regret_ave, color='blue', label=label_legend[0]+', beta={}'.format(learning_pram[0]))
    plt.plot(range(playtime), game2_Regret_ave, color='red', label=label_legend[1]+', beta={}'.format(learning_pram[1]))
    plt.plot(range(playtime), game3_Regret_ave, linestyle = "dashed", color='blue', label=label_legend[2]+', beta={}'.format(learning_pram[2]))
    plt.plot(range(playtime), game4_Regret_ave, linestyle = "dashed", color='red', label=label_legend[3]+', beta={}'.format(learning_pram[3]))
    plt.plot(range(playtime), game5_Regret_ave, linestyle="dashdot", color='blue',
             label=label_legend[4] + ', beta={}'.format(learning_pram[4]))
    plt.plot(range(playtime), game6_Regret_ave, linestyle="dashdot", color='red',
             label=label_legend[5] + ', beta={}'.format(learning_pram[5]))
    # plt.plot(range(playtime), game3_Regret_ave, color='yellow', label='beta={}'.format(learning_pram[2]))
    # plt.plot(range(playtime), game4_Regret_ave, color='green', label='beta={}'.format(learning_pram[3]))
    plt.tick_params(labelsize=12)
    plt.rcParams["font.size"] = 12
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('regret', fontsize=15)
    # plt.xlabel('play times')
    # plt.ylabel('Regret')
    # plt.xticks(np.arange(0, iteration + 1, xticksinterval))
    plt.xticks(np.arange(0, playtime+1, step=5000))
    plt.xlim([0, playtime])
    plt.ylim([0, 20000])
    # plt.legend(fontsize=fosize)
    # plt.tick_params(labelsize=lasize)
    plt.legend(loc='upper left')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comp_others.png')
    fig.savefig('comp_others.eps')
    #plt.show()