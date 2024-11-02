import os
import argparse
import torch
import torchvision

from lib.network.model import SingleLevelCILMultiFrame, targetsDetectionNet
from lib.engine.onlineSimulation import onlineSimulationWithNetwork

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_args():
    parser = argparse.ArgumentParser(description='Test the policy network in simulation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default="checkpoints_policy/policy_model.pth", 
                        help='Path of trained model for saving')
    parser.add_argument('-ldm', '--LD-model-dir', dest='LD_model_dir', type=str, default="checkpoints_LD/lumen_detection_model.pth", 
                        help='Path of trained model for saving')
    parser.add_argument('--history-length', type=int, default=10, metavar='T', help='Number of consecutive states processed')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SingleLevelCILMultiFrame(args.history_length)
    pretrained_dict = torch.load(args.model_dir, map_location=device)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
    model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
    net.load_state_dict(model_dict)
    net.to(device=device)

    target_detection_net = targetsDetectionNet()
    pretrained_dict = torch.load(args.LD_model_dir, map_location=device)
    model_dict = target_detection_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 不必要的键去除掉
    model_dict.update(pretrained_dict)  # 覆盖现有的字典里的条目
    target_detection_net.load_state_dict(model_dict)
    target_detection_net.to(device=device)

    online_test_centerline_name = 'siliconmodel3 Centerline model'

    transform_eval = torchvision.transforms.ToTensor()

    with torch.no_grad():
        net.eval()
        target_detection_net.eval()
        simulator = onlineSimulationWithNetwork(args, online_test_centerline_name, renderer='pyrender', training=False)
        simulator.run(net, target_detection_net, epoch=None, transform_func=transform_eval, training=False)
        