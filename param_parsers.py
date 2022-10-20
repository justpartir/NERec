"""Parameter parsing module."""

import argparse

def parameter_parsers():
    """

    """
    parser = argparse.ArgumentParser(description="")
    
    #训练集训练时一次训练多少行数据
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training')
    #邻居采样百分比，选取多少比例的邻居进行聚合
    parser.add_argument('--percent', type=float, default=0.2, help='neighbor sampling percent')
    #特征向量嵌入大小，分类过细即嵌入大小太大可能导致过拟合，嵌入大小过小无法充分学习到特征
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    #学习率
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    #测试集测试时一次测试多少行
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--mepochs', type=int, default=20, metavar='N', help='number of epochs to train')
    #设置训练断点，存储当前时间点下最好训练结果
    parser.add_argument('--load_from_checkpoint', type=bool, default=False, help='Load from checkpoint or not')
    #CPU还是GPU训练
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    #数据集选择
    parser.add_argument('--data', type = str, default='ciao_rank1')
    #Adam优化时的衰减率
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')

    #Social Graph Layer的所有参数设置
    parser.add_argument("--edge-path", nargs="?", default="./input/ciao_edges_16.csv", help="Edge list csv.")
    #parser.add_argument("--embedding-path", nargs="?", default="./output/ciao_AW_embedding_16.csv", help="Target embedding csv.")
    #parser.add_argument("--attention-path", nargs="?", default="./output/ciao_AW_attention_16.csv", help="Attention vector csv.")
    parser.add_argument("--dimensions", type=int, default=64, help="Number of dimensions. Default is 128.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of gradient descent iterations. Default is 200.")
    parser.add_argument("--window-size", type=int, default=5, help="Skip-gram window size. Default is 5.")
    parser.add_argument("--num-of-walks", type=int, default=80, help="Number of random walks. Default is 80.")
    parser.add_argument("--beta", type=float, default=0.5, help="Regularization parameter. Default is 0.5.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Regularization parameter. Default is 0.5.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Gradient descent learning rate. Default is 0.01.")

    #return parser.parse_args()
    return parser.parse_known_args()[0]

