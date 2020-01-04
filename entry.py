import sys, os, argparse

sys.path.append(os.path.join(os.getcwd(), 'class'))
aa=os.getcwd() #'D:\\杂货\\郭望\\diffnet-master'

from ParserConf import ParserConf
from DataUtil import DataUtil
from Evaluate import Evaluate
from diffnet import diffnet

def executeTrainModel(config_path, model_name):
    #config_path  D:\杂货\郭望\diffnet - master\conf / yelp_diffnet.ini
    #model_name  diffnet

    #print('System start to prepare parser config file...')
    conf = ParserConf(config_path)
    conf.parserConf()

    #print(conf.topk)
    #print("conf.gpu!!!!")
    #print(conf.gpu_device)  1
    
    #print('System start to load TensorFlow graph...')

    #print(model_name) diffnet
    model = eval(model_name)
    #print(model) <class 'diffnet.diffnet'>

    model = model(conf)
    #print(model) <diffnet.diffnet object at 0x000001A1957CB9B0>

    #print('System start to load data...')
    data = DataUtil(conf)
    evaluate = Evaluate(conf)

    import train as starter
    starter.start(conf, data, model, evaluate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name',default='yelp')
    parser.add_argument('--model_name', nargs='?', help='model name',default='diffnet')
    parser.add_argument('--gpu', nargs='?', help='available gpu id',default=0)


    args = parser.parse_args()

    data_name = args.data_name
    model_name = args.model_name
    device_id = args.gpu

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    config_path = os.path.join(os.getcwd(), 'conf/%s_%s.ini' % (data_name, model_name))

    executeTrainModel(config_path, model_name)
