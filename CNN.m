%% 加载数据
% vehicleDataset 是一个 dataset 数据类型，第１列是图像的相对路径，第２列是图像中小汽车的位置
% data = load('fasterRCNNVehicleTrainingData.mat');
% 提取训练集
% vehicleDataset = data.vehicleTrainingData;
% 提取图像路径
% dataDir = fullfile(toolboxdir('vision'),'visiondata');
% vehicleDataset.imageFilename = fullfile(vehicleDataset.imageFilename);
% 随机显示 9 幅图像
load matlab
%% 设置导入选项并导入数据
opts = spreadsheetImportOptions("NumVariables", 1);

% 指定工作表和范围
opts.Sheet = "Sheet1";
opts.DataRange = "A2:A36";

% 指定列名称和类型
opts.VariableNames = "imageFilename";
opts.VariableTypes = "string";

% 指定变量属性
opts = setvaropts(opts, "imageFilename", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "imageFilename", "EmptyFieldRule", "auto");

% 导入数据
lujing = readmatrix("D:\各类竞赛文件\大数据竞赛\尝试\道路坑洼位置.xlsx", opts, "UseExcel", false);
clear opts

k = randi([1, length(lujing)], 1, 9);
I = [];
for i = 1:9
 % 读取图片
 tmp = imread(lujing{k(i)});
 % 添加标识框
 tmp = insertShape(tmp, 'Rectangle', vehicleDataset.vehicle{k(i)}, 'Color', 'r','LineWidth',8);
 I{i} = mat2gray(tmp);
end
% 显示
figure
montage(I)
%% 构建 CNN
% 输入层，最小检测对象约 32×32
inputLayer = imageInputLayer([40 22 3]);
% 中间层image
% 定义卷基层参数
filterSize = [3 3];
numFilters = 32;
middleLayers = [
 % 第 1 轮，只包含 CNN 和 ReLU
 convolution2dLayer(filterSize, numFilters, 'Padding', 1)
 reluLayer()
 % 第 2 轮，包含 CNN、ReLU 和 Pooling
 convolution2dLayer(filterSize, numFilters, 'Padding', 1)
 reluLayer()
 maxPooling2dLayer(3, 'Stride',2)
 ];
% 输出层
finalLayers = [
 % 新增一个包含 64 个输出的全连接层
 fullyConnectedLayer(64)
 % 新增一个非线性 ReLU 层
 reluLayer()
 % 新增一个有两个输出的全连接层，用于判断图像是否包含检测对象
 fullyConnectedLayer(2)
 % 添加 softmax 和 classification 层
 softmaxLayer()
 classificationLayer()
 ];
% 组合所有层
layers = [
 inputLayer
 middleLayers
 finalLayers
 ];
%% 训练 CNN
% 将数据划分为两部分
% 前 60%的数据用于训练，将后面的 40%用于测试
ind = round(size(vehicleDataset,1) * 0.6);
trainData = vehicleDataset(1 : ind, :);
testData = vehicleDataset(ind+1 : end, :);
% 训练过程包括 4 步，每步都可以使用单独的参数，也可以使用同一个参数
% options = [
%  % 第 1 步，Training a Region Proposal Network (RPN)
%  trainingOptions('sgdm', 'MaxEpochs', 10,'InitialLearnRate', 1e-5,'CheckpointPath', tempdir)
%  % 第 2 步，Training a Fast R-CNN Network using the RPN from step 1
%  trainingOptions('sgdm', 'MaxEpochs', 10,'InitialLearnRate', 1e-5,'CheckpointPath', tempdir)
%  % 第 3 步，Re-training RPN using weight sharing with Fast R-CNN
%  trainingOptions('sgdm', 'MaxEpochs', 10,'InitialLearnRate', 1e-6,'CheckpointPath', tempdir)
%  % 第 4 步，Re-training Fast R-CNN using updated RPN
%  trainingOptions('sgdm', 'MaxEpochs', 10,'InitialLearnRate', 1e-6,'CheckpointPath', tempdir)
%  ];
options = trainingOptions('sgdm',...
    'MaxEpochs',7,...
    'MiniBatchSize',1,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir);
% 设置模型的本地存储
doTrainingAndEval = 1;
if doTrainingAndEval 
 % 训练 R-CNN，神经网络工具箱提供了 3 个函数
 % （1）trainRCNNObjectDetector，训练快且检测慢，允许指定 proposalFcn
% （2）trainFastRCNNObjectDetector，速度较快，允许指定 proposalFcn
 % （3）trainFasterRCNNObjectDetector，优化运行性能，不需要指定 proposalFcn
%  detector = trainFasterRCNNObjectDetector(trainData, layers, options, ...
%         'NegativeOverlapRange', [0 0.3], ...
%         'PositiveOverlapRange', [0.6 1], ...
%         'BoxPyramidScale', 1.2);
 detector = trainFasterRCNNObjectDetector(trainData, layers, options, ...
 'NegativeOverlapRange', [0 0.3], ...
 'PositiveOverlapRange', [0.6 1]);
else
 % 加载已经训练好的神经网络
 detector = pretrained.detector;
end
% load detector.mat
% clc; clear ; close all;
%% 加载数据
% data = load('fasterRCNNVehicleTrainingData.mat');
% detector = data.detector;
%% 测试结果
I = imread('potholes22.jpg');
% 运行检测器，输出目标位置和得分
[bboxes, scores] = detect(detector, I);
% 在图像上标记汽车区域
I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
figure
imshow(I) 
%% 评估训练效果
% testData = transform(testData,@(data)preprocessData(data,inputSize));
if doTrainingAndEval
 results = struct;
 for i = 1:size(testData,1)
 % 读取测试图片
 I = imread(lujing{i});
 % 运行 CNN 检测器
 [bboxes, scores, labels] = detect(detector, I);
 % 将结果保存到结构体中
 results(i).Boxes = bboxes;
 results(i).Scores = scores;
 results(i).Labels = labels;
 end
 % 将结构体转换为 table 数据类型
 results = struct2table(results);
else
 % 加载之前评估好的数据
 results = data.results;
end
% 从测试数据中提取期望的小车位置
% expectedResults = boxLabelDatastore(testData(:,2:end));
expectedResults = testData(:, 2:end);
%采用平均精确度评估检测效果
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);
% 绘制召回率-精确率曲线
figure;
plot(recall, precision);
xlabel('recall');
ylabel('precision')
grid on;
title(sprintf('Average Precision = %.2f', ap));
