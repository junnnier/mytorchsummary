import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


def get_sub_module(model):
    sub_module = OrderedDict()
    base_module_num = 0
    for model_name, model_self in model.named_modules():
        if model_name:
            num = sum(1 for _, _ in model_self.named_children())  # 包含子模块的数量
            if num:
                if isinstance(model_self, nn.Sequential):
                    sub_module[model_name] = "Sequential"
                elif isinstance(model_self, nn.ModuleList):
                    sub_module[model_name] = "ModuleList"
                else:
                    sub_module[model_name] = "UserDefined"
            else:
                base_module_num += 1
                sub_module[model_name] = ["{}-{}".format(str(model_self.__class__).split(".")[-1].split("'")[0], base_module_num), model_self]
    return sub_module


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    """
    Args:
        model: 网络模型
        input_size: 输入特征shape(没有batch维度)
        batch_size: batch大小
        device: 计算设备
        dtypes: 数据类型
    Returns: 模型参数信息
    """
    result, params_info = summary_string(model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):

    # 用于兼容多个网络输入
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # 指定每个输入使用的tensor类型，默认float32
    if dtypes is None:
        dtypes = [torch.float32]*len(input_size)

    summary_str = ''  # 保存最终输出的结果

    def register_hook(module):
        # 注册的钩子函数，参数module,input,output是系统传给hook函数的，不能修改
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]  # 模块名
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)  # 新的模块名

            # 保存输入和输出的的batch-size
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            # 统计参数量
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))  # 模块权重的参数量
                summary[m_key]["trainable"] = module.weight.requires_grad  # 是否可训练的参数
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))  # 模块偏置的参数量
            summary[m_key]["nb_params"] = params

        # 模块注册hook，并加入列表后续便于移除
        hooks.append(module.register_forward_hook(hook))

    # 创建一个batch_size为2的输入数据
    x = [torch.rand(2, *in_size).type(dtype).to(device=device) for in_size, dtype in zip(input_size, dtypes)]

    # 获取子模块
    all_sub_module = get_sub_module(model)

    # 创建属性
    summary = {}
    hooks = []

    # 对model注册hook，对其模块执行register_hook函数
    for module_name, sub_module in all_sub_module.items():
        if isinstance(sub_module, list):
            sub_module[1].apply(register_hook)

    # 前向推理
    model(*x)

    # 删除hook
    for h in hooks:
        h.remove()

    summary_str += "-" * 102 + "\n"
    line_new = "{:<40} {:>45} {:>15}".format("Name (type)", "Input Shape -> Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "=" * 102 + "\n"

    # 计算总参数的数量
    total_params = 0  # 总参数的数量
    total_output = 0  # 输出大小
    trainable_params = 0  # 可训练参数的数量

    # 打印每个模块信息
    for module_n, sub_m in all_sub_module.items():
        module_n = "    " * (len(module_n.split(".")) - 1) + module_n.split(".")[-1]  # 调整模块名
        if isinstance(sub_m, str):
            line_new = "{:<40} {:>45} {:>15}".format("{}({})".format(module_n, sub_m), "", "")
        else:
            line_new = "{:<40} {:>45} {:>15}".format("{}({})".format(module_n, sub_m[0].split("-")[0]),
                                                     "{} -> {}".format(summary[sub_m[0]]["input_shape"], summary[sub_m[0]]["output_shape"]),
                                                     "{0:,}".format(summary[sub_m[0]]["nb_params"]))
            total_params += summary[sub_m[0]]["nb_params"]
            total_output += np.prod(summary[sub_m[0]]["output_shape"])
            if "trainable" in summary[sub_m[0]]:
                if summary[sub_m[0]]["trainable"] is True:
                    trainable_params += summary[sub_m[0]]["nb_params"]
        summary_str += line_new + "\n"

    # 计算参数量大小。假设使用的是float32即 4 bytes/参数 (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4. / (1024 ** 2.))  # 转换为MB
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # *2是用于计算梯度
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * 102 + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    summary_str += "-" * 102 + "\n"
    summary_str += "Data type: {}".format(torch.float32) + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * 102 + "\n"

    return summary_str, (total_params, trainable_params)
