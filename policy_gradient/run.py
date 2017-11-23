from brain import PolicyGradient

n_features = 10
n_actions = 4

if __name__ == "__main__":

    brain = PolicyGradient(n_actions, n_features)
    brain.store_transition([1] * n_features, 1, 1.0)
    #brain.build_net()
    brain.learn()
