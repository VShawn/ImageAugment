import sys
import getopt
import os
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from torch.autograd import Variable
from onnxsim import simplify


class pth_to_onnx(object):
    '''
    将 pytorch 模型的 pth 文件转为 onnx
    '''
    @staticmethod
    def ToOnnx(pth_path: str, input_channel: int, intput_size: int, out_onnx_path: str = ""):
        # 加载模型
        model = torch.load(pth_path)
        if model is dict and 'model' in model and 'model_state_dict' in model:
            checkpoint = model
            model = checkpoint['model']  # 提取网络结构
            model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数

        # print(model)

        if out_onnx_path == "":
            out_onnx_path = os.path.join(os.path.dirname(pth_path), os.path.basename(pth_path) + ".onnx")

        # 转为 onnx
        dummy_input = Variable(torch.randn(10, input_channel, intput_size, intput_size)).cuda()
        # print(model)
        torch.onnx.export(model,
                          dummy_input,
                          out_onnx_path,
                          export_params=True,        # store the trained parameter weights inside the model file
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],   # the model's input names
                          output_names=['output'],  # the model's output names
                          opset_version=9,  # the ONNX version to export the model to
                          )

        # simplifier by https://github.com/daquexian/onnx-simplifier
        onnx_model = onnx.load(out_onnx_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # convert model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        # save sim onnx model
        out_sim_onnx_path = os.path.join(os.path.dirname(out_onnx_path), os.path.basename(out_onnx_path) + ".sim.onnx")
        onnx.save(model_simp, out_sim_onnx_path)

        session = onnxruntime.InferenceSession(out_sim_onnx_path, providers=['CUDAExecutionProvider'])
        # session = onnxruntime.InferenceSession(pretrained + ".sim.onnx", providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        orig_result = session.run([], {input_name: dummy_input.cpu().data.numpy()})
        print("sim onnx runtime out :", orig_result[0][0])
        probabilities = torch.nn.functional.softmax(torch.from_numpy(orig_result[0][0]), dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        sum = np.sum(np.exp(orig_result[0][0]))
        for i in range(top5_prob.size(0)):
            print("TOP {0}({1}): ".format(i, np.exp(orig_result[0][0][top5_catid[i]]) / sum), top5_catid[i])


if __name__ == "__main__":
    # pth_path = 'epoch_999.pth'
    pth_path = 'ModelForDemo_20220311113655_best.pth'

    # 读取输入参数
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "-h-i:c:s:", ["i=", "c=", "s="])
        if len(opts) == 0:
            print('please set args: -i <pth file path>')
            sys.exit()
        else:
            for opt, arg in opts:
                if opt == '-h':
                    print('please set args: -i <pth file path>')
                    sys.exit()
                else:
                    if opt in ("-i"):
                        pth_path = arg
                        if os.path.exists(pth_path) == False:
                            print('pth file not exist: {}'.format(pth_path))
                            sys.exit(1)
                    if opt in ("-c"):
                        channel = int(arg)
                    if opt in ("-s"):
                        size = int(arg)
    except getopt.GetoptError:
        print('please set args: -i <pth file path> -c <model input channel e.g. 3> -s <model input size e.g. 224>')
        sys.exit(2)

    pth_to_onnx.ToOnnx(pth_path, channel, size)
