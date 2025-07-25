import torch
import gym
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_episode(env,hold_score):
    state = env.reset()[0]
    states = [state]
    total_rewards = []
    is_done_1 = False
    is_done_2 = False
    while not is_done_1 and not is_done_2:
        action = 1 if state[0] < hold_score else 0
        state,reward,is_done_1,is_done_2,_ = env.step(action)
        states.append(state)
        total_rewards.append(reward)
    return states,total_rewards

def mc_prediction_first_visit(env,hold_score,gamma,n_episode):
    V = defaultdict(float)
    N = defaultdict(int)
    for episode in range(n_episode):
        states_t,rewards_t = run_episode(env,hold_score)
        return_t = 0
        G = {}
        '''
            states_t[1::-1]     就是进行一次action的逆序 state_t[1]  state_t[0]
            rewards_t[::-1]     直接对其逆序             reward_t[-1] reward_t[-2] ... reward_t[0]
            所以这段代码的意思就是第一次访问到这个state_t[1]的reward为主
            则初始状态的回报：也就是state_t[1]的回报乘以学习率 
            跟DP差不多
        '''
        for state_t,reward_t in zip(states_t[1::-1],rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[state_t] = return_t
        for state,return_t in G.items():
            if state[0] <= 21:                      #如果玩家的点数仍小于22
                V[state] += return_t
                N[state] += 1
    for state in V:
        V[state] = V[state] / N[state]
    return V

def plot_surface(X,Y,Z,title):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(X,Y,Z,rstride=1,cstride=1,
                           cmap=matplotlib.cm.coolwarm,vmin=-1.0,vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    plt.show()

def plot_blackjack_value(V):
    player_sum_range = range(12,22)
    dealer_show_range = range(1,11)
    X,Y = torch.meshgrid([torch.tensor(player_sum_range),torch.tensor(dealer_show_range)])
    values_to_plot = torch.zeros((len(player_sum_range),len(dealer_show_range),2))
    for i,player in enumerate(player_sum_range):
        for j,dealer in enumerate(dealer_show_range):
            for k,ace in enumerate([False,True]):
                values_to_plot[i,j,k] = V[(player,dealer,ace)]  #V是字典，key查询
    plot_surface(X,Y,values_to_plot[:,:,0].numpy(),'Blackjack Value Function Without Usable Ace')
    plot_surface(X, Y, values_to_plot[:, :, 1].numpy(), 'Blackjack Value Function With Usable Ace')

env = gym.make('Blackjack-v1',render_mode = 'rgb_array')
hold_score = 18
gamma = 1
n_episode = 500000

value = mc_prediction_first_visit(env,hold_score,gamma,n_episode)
print('The value function calculated by first-visit MC prediction:\n', value)
'''

The value function calculated by first-visit MC prediction:
defaultdict(<class 'float'>, {
(19, 9, True): 0.25107604017216645, (12, 10, False): -0.2384396399169039, 
(11, 10, False): 0.040090600226500565, (20, 10, True): 0.4310918774966711, (18, 8, False): 0.11318921544814185, 
(20, 10, False): 0.4437205122444394, (21, 5, False): 0.9017758046614872, (11, 5, False): 0.1691746466028272,
(19, 1, False): -0.13336820083682008, (10, 1, False): -0.23561946902654868, (15, 8, False): -0.1338609772883689,
(5, 8, False): -0.2754880694143167, (18, 10, True): -0.23391215526046988, (14, 8, False): -0.09979564032697548,
(12, 8, False): -0.03764270286948473, (20, 1, False): 0.15351506456241032, (20, 3, False): 0.6346843615494978,
(21, 10, False): 0.8977602108036891, (16, 10, False): -0.35572599841311825, (11, 1, False): -0.10004426737494466,
(20, 9, False): 0.7505315379163714, (12, 1, False): -0.3744360902255639, (9, 1, False): -0.3329048843187661, 
(20, 2, True): 0.6079691516709511, (15, 2, True): -0.0989399293286219, (17, 9, False): -0.2633321073924237, 
(15, 9, False): -0.17385489802741558, (14, 2, False): -0.1541016273663235, (14, 2, True): -0.055327868852459015, 
(21, 1, False): 0.627471383975026, (14, 1, False): -0.4119527005433046, (15, 5, False): -0.16936026936026935, 
(5, 5, False): -0.301255230125523, (19, 9, False): 0.27252081756245267, (11, 9, False): 0.1658471166741171, 
(19, 10, False): -0.022784474764869517, (9, 5, False): 0.055627425614489, (6, 5, False): -0.318796992481203, 
(21, 7, False): 0.9214876033057852, (11, 7, False): 0.2669452181987001, (18, 10, False): -0.24777683030819833, 
(8, 10, False): -0.31759483454398707, (18, 7, False): 0.4123837493881547, (13, 7, False): -0.03545654220269846, 
(18, 2, False): 0.08220889555222388, (20, 2, False): 0.6439421338155515, (8, 7, False): 0.06963562753036437, 
(18, 1, False): -0.3778944855574123, (18, 4, False): 0.19315700072797864, (8, 4, False): -0.11439688715953307, 
(16, 4, True): -0.12347826086956522, (5, 4, False): -0.23940677966101695, (19, 5, False): 0.4536623785770543, 
(14, 7, False): -0.06099195710455764, (4, 7, False): -0.1583710407239819, (21, 10, True): 0.9219326568265682, 
(14, 9, True): -0.08712121212121213, (12, 9, True): 0.049586776859504134, (20, 6, False): 0.7025339043540328, 
(13, 10, False): -0.284384858044164, (15, 2, False): -0.17755102040816326, (5, 2, False): -0.19542619542619544, 
(12, 9, False): -0.15506232897537245, (10, 9, False): 0.08482613277133826, (15, 10, False): -0.3271636675235647,
(5, 10, False): -0.3673580786026201, (18, 9, False): -0.17022826614861583, (16, 4, False): -0.21211022480058014, 
(14, 10, False): -0.27820620284995806, (19, 4, False): 0.4123764950598024, (13, 4, False): -0.10299324106855487, 
(20, 4, False): 0.6605934409161894, (9, 6, False): 0.037467700258397935, (6, 6, False): -0.2668711656441718, 
(20, 8, False): 0.8006311490625581, (17, 10, False): -0.39814474650991916, (13, 9, False): -0.18035824583075974, 
(21, 6, False): 0.8937007874015748, (11, 6, False): 0.19091326296466268, (21, 9, True): 0.9882869692532943, 
(12, 5, False): -0.10876132930513595, (21, 2, True): 0.9787759131293189, (10, 2, False): 0.057029926595143984, 
(12, 3, False): -0.09744850906855211, (8, 3, False): -0.12992125984251968, (17, 4, False): -0.22837495475931957,
(12, 4, False): -0.09301616346447088, (10, 10, False): -0.04979028548234339, (17, 6, False): -0.23372781065088757,
(16, 6, False): -0.17400644468313642, (21, 7, True): 0.9899665551839465, (21, 3, False): 0.8881266490765172, 
(17, 3, False): -0.24620637329286799, (16, 5, False): -0.25712275765036935, (13, 5, False): -0.09412142627690331, 
(15, 4, False): -0.19078498293515359, (19, 3, False): 0.425891677675033, (14, 8, True): 0.02404809619238477, 
(16, 2, False): -0.23824901327592393, (21, 1, True): 0.6810506566604128, (20, 5, False): 0.6759393063583815, 
(17, 1, False): -0.48686577905721484, (16, 10, True): -0.2592147435897436, (12, 10, True): -0.10213243546576879, 
(8, 8, False): -0.0914332784184514, (16, 9, False): -0.2675666320526134, (19, 2, False): 0.3863087248322148, 
(12, 2, False): -0.10126974295447506, (18, 6, False): 0.27075812274368233, (11, 2, False): 0.163897620116749, 
(11, 3, False): 0.19025735294117646, (7, 2, False): -0.37300743889479276, (21, 8, False): 0.9281370923161968, 
(17, 8, False): -0.22650513950073423, (15, 1, False): -0.4231152441049485, (20, 7, False): 0.7701170117011701, 
(16, 7, False): -0.10214110214110214, (15, 7, False): -0.11502909962341663, (10, 6, False): 0.14775160599571735,
(15, 6, False): -0.14242526032919045, (6, 2, False): -0.3023952095808383, (17, 7, False): -0.19127040454222852, 
(13, 2, False): -0.13409234661606578, (12, 6, False): -0.07204433497536945, (17, 5, False): -0.24537379718726868, 
(9, 10, False): -0.23587823538987465, (16, 2, True): -0.09480122324159021, (18, 8, True): 0.11673699015471167, 
(17, 8, True): -0.12006319115323855, (16, 1, False): -0.46339754816112083, (13, 1, False): -0.35824246311738295, 
(19, 10, True): -0.022415523586483774, (14, 5, False): -0.148014440433213, (13, 6, False): -0.07060653188180405, 
(20, 3, True): 0.6630872483221476, (7, 4, False): -0.2175226586102719, (20, 8, True): 0.7877551020408163, 
(9, 8, False): 0.052700065061808715, (16, 3, False): -0.23995915588835942, (21, 4, False): 0.8793565683646113, 
(7, 3, False): -0.2792494481236203, (5, 3, False): -0.18789144050104384, (17, 1, True): -0.4307228915662651, 
(16, 5, True): -0.19008264462809918, (13, 5, True): 0.026004728132387706, (9, 9, False): -0.08392511297611362, 
(8, 1, False): -0.4525096525096525, (6, 10, False): -0.37945640663607483, (5, 1, False): -0.45537757437070936, 
(17, 2, False): -0.2656866260308354, (13, 1, True): -0.2786516853932584, (19, 3, True): 0.38589211618257263, 
(15, 6, True): -0.04659498207885305, (13, 6, True): -0.004166666666666667, (14, 10, True): -0.28415841584158413, 
(14, 3, False): -0.18603154168007724, (21, 4, True): 0.9837812789620018, (16, 1, True): -0.4124203821656051, 
(21, 9, False): 0.9407646742057081, (20, 7, True): 0.7649402390438247, (18, 5, False): 0.18823236100723012, 
(14, 6, False): -0.09166945466711275, (15, 10, True): -0.2732606873428332, (7, 10, False): -0.4201461377870564, 
(7, 1, False): -0.503735325506937, (14, 4, False): -0.143462667101402, (17, 4, True): -0.15892053973013492, 
(10, 3, False): 0.10702875399361023, (19, 1, True): -0.21621621621621623, (8, 5, False): -0.07120500782472614, 
(19, 7, False): 0.6185406387951181, (9, 7, False): 0.11050477489768076, (12, 7, False): 0.010863005431502716, 
(9, 3, False): 0.01098901098901099, (17, 9, True): -0.13690476190476192, (15, 7, True): 0.019130434782608695, 
(14, 7, True): 0.0718562874251497, (19, 6, False): 0.504399585921325, (11, 4, False): 0.19049733570159857, 
(9, 4, False): -0.01906058543226685, (19, 8, False): 0.5861797162375197, (13, 7, True): 0.10344827586206896, 
(12, 7, True): 0.061855670103092786, (21, 8, True): 0.9909177820267686, (17, 2, True): -0.15053763440860216,
(21, 6, True): 0.9838479809976247, (18, 3, False): 0.14938271604938272, (21, 3, True): 0.9799904716531682, 
(18, 9, True): -0.22496371552975328, (16, 9, True): -0.1266233766233766, (10, 5, False): 0.14248297537978, 
(7, 5, False): -0.27475728155339807, (13, 8, False): -0.07832724858944573, (13, 8, True): 0.045548654244306416, 
(8, 6, False): -0.028951486697965573, (7, 6, False): -0.15489989462592202, (14, 9, False): -0.17580645161290323, 
(17, 7, True): -0.04790419161676647, (18, 7, True): 0.4202279202279202, (20, 9, True): 0.7759784075573549, 
(15, 3, False): -0.18528610354223432, (16, 8, True): -0.05641025641025641, (13, 3, False): -0.1168, 
(13, 10, True): -0.178060413354531, (6, 8, False): -0.2686781609195402, (20, 6, True): 0.7142857142857143, 
(6, 9, False): -0.28901734104046245, (6, 4, False): -0.29115853658536583, (15, 4, True): -0.05740740740740741, 
(6, 3, False): -0.28910614525139666, (18, 1, True): -0.41562064156206413, (15, 1, True): -0.44525547445255476, 
(21, 5, True): 0.9899589228662711, (7, 7, False): -0.24564994882292732, (10, 4, False): 0.1608171817705605,
(19, 4, True): 0.390625, (8, 9, False): -0.20875420875420875, (4, 6, False): -0.1762295081967213, 
(11, 8, False): 0.2621268656716418, (8, 2, False): -0.13078797725426483, (10, 8, False): 0.22271714922049, 
(12, 6, True): 0.10572687224669604, (18, 6, True): 0.27696793002915454, (17, 6, True): -0.11287758346581876, 
(18, 2, True): 0.1280120481927711, (16, 8, False): -0.18948126801152737, (9, 2, False): -0.00426829268292683, 
(12, 8, True): 0.08755760368663594, (17, 3, True): -0.18271954674220964, (14, 3, True): -0.05928853754940711, 
(10, 7, False): 0.21998942358540455, (18, 5, True): 0.2161422708618331, (14, 4, True): -0.02414486921529175, 
(12, 4, True): 0.1308411214953271, (18, 4, True): 0.18878248974008208, (4, 8, False): -0.08, 
(18, 3, True): 0.16338028169014085, (17, 5, True): -0.16907514450867053, (14, 5, True): -0.023483365949119372, 
(6, 7, False): -0.21095890410958903, (20, 1, True): 0.15113350125944586, (21, 2, False): 0.8903531892461781, 
(19, 2, True): 0.3581267217630854, (4, 4, False): -0.2056451612903226, (15, 5, True): -0.10350877192982456, 
(12, 3, True): -0.056338028169014086, (15, 9, True): -0.07692307692307693, (7, 8, False): -0.25826446280991733, 
(4, 10, False): -0.3517877739331027, (20, 5, True): 0.6636481241914618, (5, 9, False): -0.26373626373626374, 
(16, 7, True): -0.0031201248049922, (5, 7, False): -0.1504424778761062, (12, 5, True): -0.023255813953488372, 
(17, 10, True): -0.33320978502594517, (15, 8, True): -0.010526315789473684, (12, 1, True): -0.29957805907172996,
(20, 4, True): 0.6906158357771262, (19, 8, True): 0.6410614525139665, (16, 6, True): -0.08293460925039872, 
(7, 9, False): -0.35135135135135137, (14, 1, True): -0.41398865784499056, (13, 4, True): -0.0835030549898167, 
(14, 6, True): 0.01160541586073501, (19, 6, True): 0.51775956284153, (13, 3, True): -0.02510460251046025, 
(5, 6, False): -0.23110151187904968, (19, 5, True): 0.40086830680173663, (4, 9, False): -0.10548523206751055, 
(19, 7, True): 0.6147058823529412, (13, 2, True): -0.034229828850855744, (4, 5, False): -0.32677165354330706, 
(15, 3, True): -0.08495575221238938, (6, 1, False): -0.506155950752394, (13, 9, True): -0.08126410835214447, 
(4, 2, False): -0.10232558139534884, (16, 3, True): -0.12353923205342238, (12, 2, True): 0.05928853754940711, 
(4, 3, False): -0.07228915662650602, (4, 1, False): -0.5614035087719298})

进程已结束,退出代码0

'''
print('Number of states:', len(value))

'''
Number of states: 280
'''
plot_blackjack_value(value)