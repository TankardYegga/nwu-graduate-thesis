import numpy as np
import tensorflow as tf
from models.ngcf.ngcf import *
from time import time
import os
import sys
from utils.batch_test_tf import test_model
from utils.helper import early_stopping
from utils.model_utils import count_weights

def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    del_file(dir_path)


def ngcf_train_and_test(model):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    t0 = time()
    print('SMGCN training starts ...')

    # 构建一个配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
    # config.gpu_options.per_process_gpu_memory_fraction=0.40 
    # 开启一个会话并采用上述配置
    sess = tf.Session(config=config)

    total_param_nums = count_weights()
    tf.summary.text('model params num:', tf.convert_to_tensor(str(total_param_nums)))


    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for var in tf.trainable_variables():
        value = sess.run(var)
        tf.summary.histogram('var/' + var.name, value)
        # print(dir(var))
        # if var.requires_grad:
        #     tf.summary.histogram('var_grad/' + var.name,  var.grad)

    merged_summary_op = tf.summary.merge_all()
    

    t0 = time()
    print('NGCF training starts ...')

    checkpoints_dir = '/home/exp/SMGCN/SMGCN/data/ngcf/checkpoints'
    args_info_str =  str(args.lr) + "-" + str(args.epoch) + "-" + str(args.regs) + "|" +\
         str(args.sympt_threshold) + "-" + str(args.herb_threshold)  + "|" + \
             str(args.hidden_size) + str(args.weight_size) + "|" + str(args.node_dropout) + "|" + str(args.mess_dropout)
    checkpoints_dir = os.path.join(checkpoints_dir, args_info_str)
    ensure_dir(checkpoints_dir)
    checkpoints_save_path = os.path.join(checkpoints_dir, "best_perf_checkpoint.ckpt")
    saver = tf.train.Saver(max_to_keep=1)


    embeddings_dir = '/home/exp/SMGCN/SMGCN/data/ngcf/embeddings/'
    embeddings_dir = os.path.join(embeddings_dir, args_info_str)
    ensure_dir(embeddings_dir)
    sympt_embeddings_path = os.path.join(embeddings_dir, 'sympt_embeddings.npy')
    herb_embeddings_path = os.path.join(embeddings_dir, 'herb_embeddings.npy')
    mlp_layer_weight_path = os.path.join(embeddings_dir, 'mlp_layer_weight.npy')
    mlp_layer_bias_path = os.path.join(embeddings_dir, 'mlp_layer_bias.npy')


    perf_dir = '/home/exp/SMGCN/SMGCN/data/ngcf/performances/'
    perf_dir = os.path.join(perf_dir, args_info_str)
    ensure_dir(perf_dir)
    train_perf_save_path = os.path.join(perf_dir, "train_perf.txt")
    test_perf_save_path = os.path.join(perf_dir, "test_perf.txt")
    with open(train_perf_save_path, 'w') as f:
        f.write("args settings is：\n  {0}\n".format(str(args)))
    with open(test_perf_save_path, 'w') as f:
        f.write("args settings is: \n  {0}\n".format(str(args)))

    logs_dir = '/home/exp/SMGCN/SMGCN/data/ngcf/logs'
    logs_dir = os.path.join(logs_dir, args_info_str)
    train_logs_dir = os.path.join(logs_dir, 'train/')
    test_logs_dir = os.path.join(logs_dir, 'test/')
    ensure_dir(train_logs_dir)
    ensure_dir(test_logs_dir)
    train_summary_writer = tf.summary.FileWriter(train_logs_dir, graph=tf.get_default_graph())
    test_summary_writer = tf.summary.FileWriter(test_logs_dir, graph=tf.get_default_graph())

    """
    训练模型
    小批量训练
    确定每次迭代中的mini-batch的次数
    """
    n_batch = data_generator.n_train // args.batch_size + 1
    print('N BATCH is', n_batch)
    # 记录每10次迭代时在测试集上的准确度、回召率等平均值,训练集和测试集上的损失
    precision_logger, recall_logger, ndcg_logger, hit_ratio_logger = [], [], [], []
    test_loss_logger = []
    train_loss_logger = []
    # 设置初始的召回率最初值 和 初始的步骤数
    recall_best_value_pre = 0
    stopping_step = 0
    for epoch in range(args.epoch):

        t1 = time()
        # 每轮的损失
        total_loss, embed_loss, reg_loss = 0.0, 0.0, 0.0

        # 每次mini-batch更新参数
        for index in range(n_batch):
            # 从训练集中随机抽取batch_size大小的样本
            batch_sympt_set_mat, batch_herb_set_mat, sympt_set_normalized_mat = model.data_generator.sample_for_train()

            # 本轮会话运行损失函数loss和优化求解器opt
            # 需要馈入的参数有
            _, batch_total_loss, batch_embed_loss, batch_reg_loss, summary = sess.run([model.opt, model.total_loss, model.embed_loss,
                                                                              model.reg_loss, merged_summary_op],
                                                                             feed_dict={
                                                                                 model.sympt_set_sample_mat:batch_sympt_set_mat,
                                                                                 model.herb_set_sample_mat:batch_herb_set_mat,
                                                                                 model.sympt_set_normalized_mat:sympt_set_normalized_mat
                                                                             })
            total_loss += batch_total_loss
            embed_loss += batch_embed_loss
            reg_loss += batch_reg_loss
            train_summary_writer.add_summary(summary, epoch * n_batch + index)

        # 判断损失函数计算过程中是否出现错误
        if np.isnan(total_loss) == True:
            print('Error:Loss is nan')
            sys.exit()

        # 打印迭代次数，本论迭代的时间、总损失、嵌入损失和正则化损失
        train_performance_str = 'Epoch %d [time=%.1fs]: total_loss=%.5f, embed_loss=%.5f, reg_loss=%.5f\n ' \
                                % ( epoch, time()-t1, total_loss, embed_loss, reg_loss)
        print(train_performance_str)
        with open(train_perf_save_path, 'a+') as f:
            f.write("{0}\n\n ".format(train_performance_str))
        
        # 保留本次模型中的嵌入和训练参数
        whole_sympt_embeddings = sess.run(model.sympt_embeddings)
        whole_herb_embeddings = sess.run(model.herb_embeddings)
        # mlp_layer_weight = sess.run(model.weights['MLP_W'])
        # mlp_layer_bias = sess.run(model.weights['MLP_b'])

        """ 每10次迭代后在测试集上测试一下模型的当前效果 """
        if (epoch + 1) % 10 ==0:

            t2 = time()

            # 在测试集上测试模型,返回相应的测量指标
            test_result, test_loss = test_model(sess, model, model.data_generator, model.args, 
                            merged_summary_op, test_summary_writer)
            train_loss_logger.append(total_loss)
            test_loss_logger.append(test_loss)
            precision_logger.append(test_result['precision'])
            recall_logger.append(test_result['recall'])
            ndcg_logger.append(test_result['ndcg'])
            hit_ratio_logger.append(test_result['hit_ratio'])
            # 输出测试结果
            test_performance_str = 'Epoch %d [time=%.1fs + %.1fs]:\n ' \
                                    'train:total_loss=%.5f, embed_loss=%.5f, reg_loss=%.5f \n' \
                                    'test: precision=[%s], recall=[%s], ndcg=[%s],test_loss=%.5f\n' % \
                                    (epoch, t2-t1, time()-t2, total_loss, embed_loss, reg_loss,
                                    '\t'.join(['%.5f' % r for r in test_result['precision']]),
                                    '\t'.join(['%.5f' % r for r in  test_result['recall']]),
                                        '\t'.join(['%.5f' % r for r in test_result['ndcg']]),
                                    test_loss
                                    )
            print(test_performance_str)
            with open(test_perf_save_path, 'a+') as f:
                f.write("{0}\n\n ".format(test_performance_str))

            """测试结果的某个指标在连续一定次数没有提升时，停止迭代过程"""
            recall_best_value_pre, stopping_step, should_stop = early_stopping(test_result['recall'][0],
                                                                            recall_best_value_pre, stopping_step)
            if should_stop == True:
                break
            if stopping_step == 0:
                saver.save(sess=sess, save_path=checkpoints_save_path, global_step=epoch)
                print('**************Checkpoints Saved !*************')
                np.save(sympt_embeddings_path, whole_sympt_embeddings)
                np.save(herb_embeddings_path, whole_herb_embeddings)
                # np.save(mlp_layer_weight_path, mlp_layer_weight)
                # np.save(mlp_layer_bias_path, mlp_layer_bias)
                print('***************Embeddings Saved !**************')


    #将列表转化为数组
    pres = np.array(precision_logger)
    recalls = np.array(recall_logger)
    ndcgs = np.array(ndcg_logger)
    train_losses = np.array(train_loss_logger)
    test_losses = np.array(test_loss_logger)
    # 找出测试结果中召回率最好的那一次迭代
    max_recall = max(recalls[:,0])
    max_rcl_idx = list(recalls[:,0]).index(max_recall)
    # 打印相关信息
    final_performance_str = 'Best Iter (Epoch %d):\n total_time: %.1fs \n recalls=[%s], precisions=[%s],' \
                           'ndcgs=[%s]' % (max_rcl_idx, time()-t0,
                                           '\t'.join(['%.5f' % r for r in recalls[max_rcl_idx]]),
                                           '\t'.join(['%.5f' % r for r in pres[max_rcl_idx]]),
                                           '\t'.join(['%.5f' % r for r in ndcgs[max_rcl_idx]]))
    print(final_performance_str)
    with open(test_perf_save_path, 'a+') as f:
        f.write("{0}\n\n ".format(final_performance_str))


if __name__ == "__main__":
    from utils.load_data import Data
    from models.ngcf.ngcf_parser import parse_args

    args = parse_args()
    print('Total Epochs is', args.epoch)
    
    data_generator = Data(path=args.data_path, batch_size=args.batch_size)
    print('The data has been obtained!')

    model = NGCF(data_generator=data_generator, args=args)
    ngcf_train_and_test(model)


