import paddle.fluid as fluid
import numpy as np
import paddle
import os
import time
import datetime
import utils
from args import *

def set_zero(var_name,scope=fluid.global_scope(),place=fluid.CPUPlace(),param_type="int64"):
    """
    Set tensor of a Variable to zero.
    Args:
        var_name(str): name of Variable
        scope(Scope): Scope object, default is fluid.global_scope()
        place(Place): Place object, default is fluid.CPUPlace()
        param_type(str): param data type, default is int64
    """
    param = scope.var(var_name).get_tensor()
    param_array = np.zeros(param._get_dims()).astype(param_type)
    param.set(param_array, place)
    
def share_bottom(feature_size=499,bottom_size=117,tower_nums=2,tower_size=8):
    a_data = fluid.data(name="a", shape=[-1, feature_size], dtype="float32")
    label_income = fluid.data(name="label_income", shape=[-1, 2], dtype="float32", lod_level=0)
    label_marital = fluid.data(name="label_marital", shape=[-1, 2], dtype="float32", lod_level=0)
    
    #499*8*16 + 2*(16*8 + 8*2) = 64160 
    #64160 / (499 + 2*(8 + 8*2)) = 117
    

    bottom_output = fluid.layers.fc(input=a_data,
                                       size=bottom_size,
                                       act='relu',
                                       bias_attr=fluid.ParamAttr(learning_rate=1.0),
                                       name='bottom_output')
  
   
    # Build tower layer from bottom layer
    output_layers = []
    for index in range(tower_nums):    
        tower_layer = fluid.layers.fc(input=bottom_output,
                                   size=tower_size,
                                   act='relu',
                                   name='task_layer_' + str(index))
        output_layer = fluid.layers.fc(input=tower_layer,
                                   size=2,
                                   act='softmax',
                                   name='output_layer_' + str(index))
        output_layers.append(output_layer)

    cost_income = paddle.fluid.layers.cross_entropy(input=output_layers[0], label=label_income,soft_label = True)
    cost_marital = paddle.fluid.layers.cross_entropy(input=output_layers[1], label=label_marital,soft_label = True)
    

    label_income_1 = fluid.layers.slice(label_income, axes=[1], starts=[1], ends=[2])
    label_marital_1 = fluid.layers.slice(label_marital, axes=[1], starts=[1], ends=[2])
    
    auc_income, batch_auc_1, auc_states_1  = fluid.layers.auc(input=output_layers[0], label=fluid.layers.cast(x=label_income_1, dtype='int64'))
    auc_marital, batch_auc_2, auc_states_2 = fluid.layers.auc(input=output_layers[1], label=fluid.layers.cast(x=label_marital_1, dtype='int64'))
    
    avg_cost_income = fluid.layers.mean(x=cost_income)
    avg_cost_marital = fluid.layers.mean(x=cost_marital)
    
    cost =  avg_cost_income + avg_cost_marital
    
    return [a_data,label_income,label_marital],cost,output_layers[0],output_layers[1],label_income,label_marital,auc_income,auc_marital,auc_states_1,auc_states_2



args = parse_args()
train_path = args.train_data_path
test_path = args.test_data_path
batch_size = args.batch_size
feature_size = args.feature_size
bottom_size = args.bottom_size
tower_nums = args.tower_nums
tower_size = args.tower_size
epochs = args.epochs

print("batch_size:[%d],epochs:[%d],feature_size:[%d],bottom_size:[%d],tower_nums:[%d],tower_size:[%d]"%(batch_size,epochs,feature_size,bottom_size,tower_nums,tower_size))

train_reader = utils.prepare_reader(train_path,batch_size)
test_reader = utils.prepare_reader(test_path,batch_size)
  
data_list,loss,out_1,out_2,label_1,label_2,auc_income,auc_marital,auc_states_1,auc_states_2 = share_bottom(feature_size,bottom_size,tower_nums,tower_size)     


Adam = fluid.optimizer.AdamOptimizer()
Adam.minimize(loss)
place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
test_program = fluid.default_main_program().clone(for_test=True)


loader = fluid.io.DataLoader.from_generator(feed_list=data_list, capacity=batch_size, iterable=True)
loader.set_sample_list_generator(train_reader, places=place)

test_loader = fluid.io.DataLoader.from_generator(feed_list=data_list, capacity=batch_size, iterable=True)
test_loader.set_sample_list_generator(test_reader, places=place)
auc_income_list = []
auc_marital_list = []

for epoch in range(epochs):
    begin = time.time()
    for var in auc_states_1:  # reset auc states
        set_zero(var.name,place=place)
    for var in auc_states_2:  # reset auc states
        set_zero(var.name,place=place)
    begin = time.time()
    auc_1_p = 0.0
    auc_2_p = 0.0
    loss_data =0.0
    for batch_id,train_data in enumerate(loader()):
        
        loss_data,out_income,out_marital,label_income,label_marital,auc_1_p,auc_2_p = exe.run(
                  feed=train_data,
                  fetch_list=[loss.name,out_1,out_2,label_1,label_2,auc_income,auc_marital],
                  return_numpy=True)
    
    for var in auc_states_1:  # reset auc states
        set_zero(var.name,place=place)
    for var in auc_states_2:  # reset auc states
        set_zero(var.name,place=place)    
    test_auc_1_p = 0.0
    test_auc_2_p = 0.0
    for batch_id,test_data in enumerate(test_loader()):
        
        test_out_income,test_out_marital,test_label_income,test_label_marital,test_auc_1_p,test_auc_2_p = exe.run(
                  program=test_program,
                  feed=test_data,
                  fetch_list=[out_1,out_2,label_1,label_2,auc_income,auc_marital],
                  return_numpy=True) 
    model_dir = os.path.join(args.model_dir,'epoch_' + str(epoch + 1), "checkpoint")
    main_program = fluid.default_main_program()
    fluid.io.save(main_program,model_dir)
    
    auc_income_list.append(test_auc_1_p)
    auc_marital_list.append(test_auc_2_p)
    end = time.time()
    time_stamp = datetime.datetime.now()
    print("%s,- INFO - epoch_id: %d,epoch_time: %.5f s,loss: %.5f,train_auc_income: %.5f,train_auc_marital: %.5f,test_auc_income: %.5f,test_auc_marital: %.5f"%
    (time_stamp.strftime('%Y-%m-%d %H:%M:%S'),epoch,end - begin,loss_data,auc_1_p,auc_2_p,test_auc_1_p,test_auc_2_p))
    
time_stamp = datetime.datetime.now()
print("%s,- INFO - mean_sb_test_auc_income: %.5f,mean_sb_test_auc_marital %.5f,max_sb_test_auc_income: %.5f,max_sb_test_auc_marital %.5f"%(
    time_stamp.strftime('%Y-%m-%d %H:%M:%S'),np.mean(auc_income_list),np.mean(auc_marital_list),np.max(auc_income_list),np.max(auc_marital_list)))    
        
        
        
        
        
        
        
        
