import numpy as np
import pickle
import rlglue
import environment
import agents
import plot_utils


class Options(object):

    def __init__(self, env, alpha=0.1, epsilon=1.0, discount=0.1):

        # Configuring Environment
        self.env = env
        self.env.set_goal_state((-1, -1)) # (-1, -1) implies no goal_state
        self.env.set_start_state((-1, -1)) # no start_state
        self.env.add_terminate_action()
        self.max_row, self.max_col = self.env.get_grid_dimension()

        # Configuring Agent
        self.agent = agents.QAgent(self.max_row, self.max_col)
        self.agent.add_terminate_action()
        self.agent.set_alpha(alpha)
        self.agent.set_epsilon(epsilon)
        self.agent.set_discount(discount)

        self.glue = rlglue.RLGlue(self.env, self.agent)

        self.eigenvectors = None
        self.eigenoptions = []
        self.option_idx = 0
        # compute eigen
        self.compute_eigen()

    def compute_eigen(self):

        # Need to exclude Terminate Action
        default_max_actions = self.env.get_default_max_actions()

        # get all possible (r,c) states in env
        states_rc = []
        for r in range(self.max_row):
            for c in range(self.max_col):
                states_rc.append((r, c))

        total_states = len(states_rc)

        #-- Compute adjacency matrix (take all possible actions from every state)
        adjacency = np.zeros((total_states, total_states), dtype = np.int)

        for state in range(total_states):

            if self.env.states_rc[state] in self.env.obstacle_vector:
                continue
        # Jonas flipped these: starting over again many times and not walking to far seems to give something that looks more like a kernel..
            for i in range(1000):
                self.env.set_current_state(state)
                for j in range(5):

                    result = self.env.step(np.random.choice(range(4)))
                    if result['state'] is not None:
                        next_state = result['state'][0]

                        adjacency[state][next_state] += 1
                    else:
                        adjacency[state][next_state] += 1
                        break

        adj_diag = 1000*np.eye(total_states) + adjacency
        #-- Creating a kernel matrix out of the random walk adjacency graph
        def check_symmetric(a, rtol=1e-05, atol=1e-08):
            return np.allclose(a, a.T, rtol=rtol, atol=atol)

        print(check_symmetric(adj_diag))
        adj_sym = adj_diag +adj_diag.T

        print(check_symmetric(adj_sym))
        dis_mat = (adj_sym.max(1, keepdims=True)-adj_sym)/(adj_sym.max(1, keepdims=True)+0.0001)

        # kernel_size = 0.1 # <- Gives bullshait
        kernel_size = 0.45
        # kernel_size = 2 # <- Similar to 0.45, but more smoothing
        K = np.exp(-0.5*(dis_mat**2/kernel_size**2))
        #--

        from scipy import linalg
        w, v = linalg.eigh(K)


        # extract eigenvalues(w), eigenvectors(v)
        #w, v = np.linalg.eigh(L)

        #idx_s = np.flip(np.argsort(w))
        #w = w[idx_s]
        #v = v[:,idx_s]
        v_sum = np.dot(v.T, np.ones_like(w))
        #v_sum = v.sum(0)
        scores = (np.sqrt(np.abs(w))*v_sum)**2

        #---IT---
        indexes = np.flip(np.argsort(scores))
        #eigenvectors = v[indexes]

        #---OG---: sort in order of increasing eigenvalue
        #indexes = np.argsort(w)

        # self.eigenoptions will be computed lazily

        #indexes = np.argsort(w)
        #eigenvalues = w[indexes]
        self.eigenvectors = v[:,indexes]
        #self.eigenvectors = v.T
        #self.eigenvectors = np.concatenate((self.eigenvectors[:,1:], np.zeros((100,1))), axis=1)
        #eigenvectors = v[indexes,:]

        # sort in order of increasing eigenvalue
        # self.eigenoptions will be computed lazily

        # # Adding eigenvectors in the opposite directions
        shape = self.eigenvectors.shape
        shape = (shape[0] * 2, shape[1])
        eigenvectors = np.zeros(shape)
        
        for idx in range(len(self.eigenvectors)):
            v1 = self.eigenvectors[idx] * 1
            # v2 is the opposite eigenvector
            v2 = self.eigenvectors[idx] * -1
            eigenvectors[idx*2] = v1
            eigenvectors[idx*2 + 1] = v2


        self.eigenvectors = eigenvectors

    def learn_next_eigenoption(self, steps=100000):

        # learn next option
        if self.option_idx == len(self.eigenvectors):
            print("All eigenoptions have already been computed")
            return
        #print self.eigenvectors[self.option_idx]
        # set reward vector
        self.env.set_eigen_purpose(self.eigenvectors[self.option_idx])

        # Learn policy
        while steps >= 0:
            is_terminal = self.glue.episode(steps)
            if is_terminal is True:
                ep_steps = self.agent.get_steps()
            else:
                break
            steps -= ep_steps

        eigenoption = self.agent.get_policy()
        self.eigenoptions.append(eigenoption)
        self.option_idx += 1
        self.glue.cleanup() # reset Q(S,A) and reward vector

        # return newly learned policy
        return self.eigenoptions[-1]

    def get_eigenoptions(self):        
        return self.eigenoptions

    # display eigenoption at the idx
    def display_eigenoption(self, display = True, savename='', idx = -1):
        # default return latest learned eigenoption
        if len(self.eigenoptions) < 1 or idx not in range(-1, len(self.eigenoptions)):
            print("The eigenoption has not been learnt for this option yet")
            return

        plot_utils.plot_pi(self.eigenoptions[idx], self.max_row,
                           self.max_col, display, savename)
