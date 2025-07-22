import torch
import matplotlib.pyplot as plt
'''
Initializes the policy values as all zeros. Updates the values based on the Bellman expectation equation. 
Computes the maximal change of the values across all states. 
If the maximal change is greater than the threshold, it keeps updating the values. 
Otherwise, it terminates the evaluation process and returns the latest values.
'''
def policy_evaluation(policy,trans_matrix,rewards,gamma,threshold):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state,actions in enumerate(policy):
            for action,probs in enumerate(actions):
                V_temp[state] += probs * (rewards[state] + gamma * (torch.dot(trans_matrix[state,action],V)))
                '''
                    torch.dot 对公式后半部分做点积
                    也可以再对此写个循环
                    
                    V_new_state = 0
                    for i,prob_new_state in enumerate(trans_matrix[state,action]):
                        V_new_state += prob_new_state * V[i]
                    V_temp[state] += probs * (rewards[state] + gamma * V_new_state)
                    
                    其实就是点积，优化了代码
                '''
        delta_max = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if delta_max <= threshold:
            break
    return V

#记录历史V
def policy_evaluation_history(policy, trans_matrix, rewards, gamma, threshold):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    V_his = [V]
    while True:
        V_temp = torch.zeros(n_state)
        for state, actions in enumerate(policy):
            for action, probs in enumerate(actions):
                V_temp[state] += probs * (rewards[state] + gamma * (torch.dot(trans_matrix[state, action], V)))
        delta_max = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        V_his.append(V)
        if delta_max <= threshold:
            break
    return V,V_his



T = torch.tensor([[[0.8, 0.1, 0.1],
                   [0.1, 0.6, 0.3]],
                  [[0.7, 0.2, 0.1],
                   [0.1, 0.8, 0.1]],
                  [[0.6, 0.2, 0.2],
                   [0.1, 0.4, 0.5]]]
                )
R = torch.tensor([1.,0.,-1.])
gamma = 0.5 #0.2 0.5 0.99
threshold = 1e-4
policy_optimal = torch.tensor(
    [[1.0,0.0],
     [1.0,0.0],
     [1.0,0.0]]
)
#V = policy_evaluation(policy_optimal, T, R, gamma, threshold)
V,V_his = policy_evaluation_history(policy_optimal,T,R,gamma,threshold)

s0, = plt.plot([v[0] for v in V_his])
s1, = plt.plot([v[1] for v in V_his])
s2, = plt.plot([v[2] for v in V_his])
plt.title('Optimal policy with gamma = {}'.format(str(gamma)))
plt.xlabel('Iteration')
plt.ylabel('Policy values')
plt.legend([s0,s1,s2],
           ["State s0",
            "State s1",
            "State s2"]
           ,loc="upper left")
plt.show()


print("The value function under the optimal policy is:\n{}".format(V))
'''
The value function under the optimal policy is:
tensor([ 1.6786,  0.6260, -0.4821])
'''