#数据存储流程

    1. 程序开始时，应该保存环境初始化后的状态，作为起始状态。
    也就是需要在storage中的第一个step的位置保存初始化的[obs, masks, hidden_states],
    其中，masks用来确定当前状态是否为结束状态；hidden_state应该为当前状态与上一个状态通过
    gru得到的hidden_state, 在程序刚开始时，由于obs为第一个状态，所以其对应的hidden_states
    应该是0与当前obs得到的；
    
    2. 当程序运行时， 当前的hidden_states应该是通过当前obs与上一个hidden_states通过gru
    得到的；
    
    注意： 当存储数据至storage中时， 对storage的操作需要分为两种情况
    (1) 下一个状态为正常状态（非终止），则正常的采样-存储
    (2) 下一个状态为终止状态[此时需要对环境进行重置]，