load 'd:\test\actions_train.mat';
train_x=datas*100;
train_y=zeros(size(train_x,1),10);
for i=1:size(train_x,1)
    train_y(i,labels(i))=1;
end
nn=nnsetup([1000 500 10]);
%nn.weightPenaltyL2 = 1e-4;
nn.weightPenaltyL2 = 1e-4; 
opts.numepochs =  100;   %  Number of full sweeps through data
opts.batchsize = 40;  %  Take a mean gradient step over this many samples
[nn,L]=nntrain(nn,train_x,train_y,opts);
load 'd:\test\actions_test.mat';
test_x=datas*100;
test_y=zeros(size(test_x,1),10);
for i=1:size(test_x,1)
    test_y(i,labels(i))=1;
end
[er,bad]=nntest(nn,test_x,test_y);
er
assert(er < 0.4, 'Too big error');