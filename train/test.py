from ofa.model_zoo import proxylessnas_mobile
import torchvision.models as models
model = proxylessnas_mobile()
model1 = models.mobilenet_v2()
# for m in model.classifier.modules():
#     for param in m.parameters():
#         print(param)
# # print(model.classifier.parameters())

# def set_module_grad_status(module, flag=False,list_param = []):
#     if isinstance(module, list):
#         for m in module:
#             set_module_grad_status(m, flag)
#             return list_param
#     else:
#         for p in module.parameters():
#             list_param.append(p)
# # list_param = set_module_grad_status(model)
# print(list_param)
#print(model1.parameters)
for param in model.parameters():
    param.requires_grad =False
    print(param)
    print("----------------------")
for param in model.classifier.parameters():
    param.requires_grad =True
    print(param)
