clc;clear;close all
tic
load('02_LABEL.mat')
load('01_feature_vgg16.mat')
len=length(label);
idx = randperm(len); 
num_train = round(0.7*len); 
num_val = round(0*len); 
x_train=feature_all(idx(1:num_train),:);
x_val=feature_all(idx(num_train+1:num_train+num_val),:);
x_test=feature_all(idx(num_train+num_val+1:end),:);
Label = categorical(label);
Label_Train = Label(idx(1:num_train));
Label_Val = Label(idx(num_train+1:num_train+num_val));
Label_Test = Label(idx(num_train+num_val+1:end));
SVM_3=SVM_3_Classifier(x_train, Label_Train);
delete('SVM.mat')
save('SVM.mat',"SVM_3")
Yp_Train = SVM_3.predictFcn(x_train);
figure
plotconfusion(Label_Train,Yp_Train)
title('训练集')
[A,~]=confusionmat(Label_Train,Yp_Train);
A=A';
Acc=(A(1,1)+A(2,2))/sum(A,"all");
Precise=A(2,2)/(A(2,1)+A(2,2));
Recall=A(2,2)/(A(1,2)+A(2,2));
F1_score=2*Precise*Recall/(Precise+Recall);
disp(['训练集分类准确率：',num2str(Acc*100),'%'])
disp(['训练集分类查准率：',num2str(Precise*100),'%'])
disp(['训练集分类查全率：',num2str(Recall*100),'%'])
disp(['训练集分类F1值：',num2str(F1_score*100),'%'])
disp('------------------------------------------')
Yp_Test = SVM_3.predictFcn(x_test);
figure
plotconfusion(Label_Test,Yp_Test)
title('测试集')
[A,~]=confusionmat(Label_Test,Yp_Test);
A=A';
Acc=(A(1,1)+A(2,2))/sum(A,"all");
Precise=A(2,2)/(A(2,1)+A(2,2));
Recall=A(2,2)/(A(1,2)+A(2,2));
F1_score=2*Precise*Recall/(Precise+Recall);
disp(['测试集分类准确率：',num2str(Acc*100),'%'])
disp(['测试集分类查准率：',num2str(Precise*100),'%'])
disp(['测试集分类查全率：',num2str(Recall*100),'%'])
disp(['测试集分类F1值：',num2str(F1_score*100),'%'])
toc

