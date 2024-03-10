clc;clear;close all
image_path =[pwd '\testdata_V2\'] ;
namelist=dir([image_path,'*.jpg']);
num_data=size(namelist,1);
tic
load('vgg16.mat')
inputSize = net.Layers(1).InputSize;
layer = 'fc6';
feature_all=[];
ind_error=0;
for i=1:num_data
    try
        I =imread([image_path namelist(i).name]);
        I =imresize(I,[224 224]);
        feature= activations(net,I,layer,'OutputAs','rows');
        feature_all=cat(1,feature_all,feature);
        file(i).name=namelist(i).name;
    catch
        ind_error=ind_error+1;
        error_file(ind_error).name=namelist(i).name;%错误文件
        continue
    end
end
feature_all=double(feature_all);
load('SVM.mat')
Yp = SVM_3.predictFcn(feature_all);
toc
columns = {'fnames', 'label'};
data = table({file.name}', Yp, 'VariableNames', columns);
writetable(data, '特征提取.csv')


