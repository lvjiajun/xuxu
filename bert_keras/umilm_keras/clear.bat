echo 删除tensorboard 日志
cd logs
del *
echo 退出根目录
cd ../
echo 删除log日志 日志
del default.log
echo 删除vallog日志 日志
cd vallog
del *
echo 退出根目录
cd ../
echo 删除所有训练结果
cd save_model
del *
echo 退出根目录
cd ../
echo 删除预测结果
cd text
del *
echo 退出根目录
cd ../