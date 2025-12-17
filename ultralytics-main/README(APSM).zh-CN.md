APSM单独的模块存放在apsm\_works/apsm\_module/apsm\_module.py中。



当然我们已经把这个模块集成到了ultralytics的block中，可以直接修改apsm\_works/model\_tools下面的训练脚本来选择用哪种模型以及是否集成APSM进行训练。



但是在进行验证之前要先使用apsm\_works/apsm\_tools/add\_noise.py构建一个用于验证的噪声数据集。



然后修改apsm\_works/model\_tools下面的验证脚本来进行有无噪声的对比验证测试。



配置好的集成了APSM的模型配置文件位于apsm\_works/model\_config文件夹下。

