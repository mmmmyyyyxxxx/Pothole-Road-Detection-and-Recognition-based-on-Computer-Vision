clc
clear
load('net301.mat');
all_pic = dir( 'C:\Users\85793\Documents\MATLAB\Examples\R2022a\nnet\TransferLearningUsingGoogLeNetExample\测试\testdata_V2\*.jpg');
pic_name = cat(1,all_pic.name);
% 设置输入图像的文件夹
input_folder = 'testdata_V2';
 
% 获取文件夹中的所有图像文件
files = dir(fullfile(input_folder, '*.jpg'));
 C=[];
 B=[];
tic
% 循环读取每一张图像并导出
for i = 1:length(files)
   file = files(i).name;
   % 读取图像
   I = imread(fullfile(input_folder, file),'jpg');
   % 设置输出文件名
%    output_file = ['output_', fi
I = imresize(I, [224 224]);
[YPred,probs] = classify(net,I);
C=[C,YPred;];
B=[B,probs];

end
toc
TB=B';
tc=C';
% %% 导入数据
% %图片路径
% image_path =[pwd '\testdata_V2\'] ;%pwd读取当前路径，注意路径中不要有中文
% %将图像的文件夹路径读入
% namelist=dir([image_path,'*.jpg']);
% num_data=size(namelist,1);