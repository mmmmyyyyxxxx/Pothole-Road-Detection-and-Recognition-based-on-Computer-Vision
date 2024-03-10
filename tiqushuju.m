clc;clear;close all
image_path =[pwd '\data\'] ;
namelist=dir([image_path,'*.jpg']);
num_data=size(namelist,1);
for i=1:size(namelist,1)
    name=namelist(i).name;
    if strcmp(name(1:6),'normal')
        label(i,:)=0;
    else
        label(i,:)=1;
    end
end
load('vgg16.mat')
inputSize = net.Layers(1).InputSize;
layer = 'fc6';
feature_all=[];
ind_error=0;
for i=1:num_data
    try
        I =imread([image_path namelist(i).name]);%
        I =imresize(I,[224 224]);
        feature= activations(net,I,layer,'OutputAs','rows');
        feature_all=cat(1,feature_all,feature);
    catch
        ind_error=ind_error+1;
        error_file(ind_error).name=namelist(i).name;
        label(i)=[];
        continue
    end
end
feature_all=double(feature_all);
delete('shuju.mat')
save('shuju.mat',"feature_all")
delete('biaoqian.mat')
save('biaoqian.mat',"label")


