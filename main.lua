require 'cutorch'
require 'cunn'
require 'nnx'
require 'image'
require 'optim'
require 'cudnn'

--commandline options
opt = lapp[[
-n,--network        (default "")    reload pretrained network
-r,--learningRate   (default 10)    learning rate
-b,--batchSize      (default 1)     batch size
-m,--momentum       (default 0)     momentum for SGD
]]


input = torch.Tensor(60,1,500,500)
output = torch.Tensor(60,1,500,500)
--load stack 4
for x = 0,29 do
    local index = x+1
    local inputT = torch.Tensor(1, 500, 500)
    inputT[1] = image.load('data/stack04/04_raw-'..x..'.png')
    input[index] = inputT
    local outputT = torch.Tensor(1, 500, 500)
    outputT[1] = image.load('data/stack04/04_lbl-'..x..'.png')
    output[index] = outputT 
end
--load stack 6
for x = 0,29 do
    local index = x+31
    local inputT = torch.Tensor(1, 500, 500)
    inputT[1] = image.load('data/stack06/06_raw-'..x..'.png')
    input[index] = inputT
    local outputT = torch.Tensor(1, 500, 500)
    outputT[1] = image.load('data/stack06/06_lbl-'..x..'.png')
    output[index] = outputT 
end
output:apply(function(x)
    if x == 0 then
        return 0
    else 
        return 1
    end
end)

output = output:cuda()
input = input:cuda()


--fullsize branch
fullm = nn.Sequential()

b0 = nn.Sequential()
b0:add(nn.SpatialConvolutionMM(1,64,3,3))
b0:add(nn.SpatialConvolutionMM(64,64,3,3))
b0:add(nn.ReLU())
fullm:add(b0)
--branch 1
b1 = nn.Sequential()
b1:add(nn.SpatialMaxPooling(2,2,2,2))
b1:add(nn.SpatialConvolutionMM(64,128,3,3))
b1:add(nn.SpatialConvolutionMM(128,128,3,3))
b1:add(nn.ReLU())
--create skip
con1 = nn.ConcatTable()
con1:add(b1)
con1:add(nn.Identity())
fullm:add(con1)

--branch 2
b2 = nn.Sequential()
b2:add(nn.SpatialMaxPooling(2,2,2,2))
b2:add(nn.SpatialConvolutionMM(128,256,3,3))
b2:add(nn.SpatialConvolutionMM(256,256,3,3))
b2:add(nn.ReLU())
con2 = nn.ParallelTable()
con2c = nn.ConcatTable()
con2c:add(b2)
con2c:add(nn.Identity())
con2:add(con2c)
con2:add(nn.Identity())
con2t = nn.Sequential()
con2t:add(con2)
con2t:add(nn.FlattenTable())
fullm:add(con2t)


--branch 3
b3 = nn.Sequential()
b3:add(nn.SpatialMaxPooling(2,2,2,2))
b3:add(nn.SpatialConvolutionMM(256,512,3,3))
b3:add(nn.SpatialConvolutionMM(512,512,3,3))
b3:add(nn.SpatialConvolutionMM(512,512,3,3))
b3:add(nn.ReLU())
con3 = nn.ParallelTable()
con3c = nn.ConcatTable()
con3c:add(b3)
con3c:add(nn.Identity())
con3:add(con3c)
con3:add(nn.Identity())
con3:add(nn.Identity())
con3t = nn.Sequential()
con3t:add(con3)
con3t:add(nn.FlattenTable())
fullm:add(con3t)

b4 = nn.Sequential()
b4:add(nn.SpatialMaxPooling(2,2,2,2))
b4:add(nn.SpatialConvolutionMM(512,512,3,3))
b4:add(nn.SpatialConvolutionMM(512,512,3,3))
b4:add(nn.SpatialConvolutionMM(512,512,3,3))
b4:add(nn.ReLU())
con4 = nn.ParallelTable()
con4c = nn.ConcatTable()
con4c:add(b4)
con4c:add(nn.Identity())
con4:add(con4c)
con4:add(nn.Identity())
con4:add(nn.Identity())
con4:add(nn.Identity())
con4t = nn.Sequential()
con4t:add(con4)
con4t:add(nn.FlattenTable())
fullm:add(con4t)


post = nn.ParallelTable()
--handles 512 layers into upsampled 500x500
post1 = nn.Sequential()
post1:add(nn.SpatialFullConvolution(512,256,3,3,5,5,0,0,2,2))
post1:add(nn.SpatialFullConvolution(256,1,3,3,5,5,0,0,2,2))
post1:add(nn.ReLU())
post1:add(nn.Mul())
post:add(post1)
post2 = nn.Sequential()
post2:add(nn.SpatialFullConvolution(512,256,3,3,2,2,1,1,1,1))
post2:add(nn.SpatialFullConvolution(256,1,3,3,5,5,14,14))
post2:add(nn.ReLU())
post2:add(nn.Mul())
post:add(post2)
post3 = nn.Sequential()
post3:add(nn.SpatialFullConvolution(256,128,3,3,5,5,20,20))
post3:add(nn.SpatialFullConvolution(128,1,3,3,1,1,25,25))
post3:add(nn.ReLU())
post3:add(nn.Mul())
post:add(post3)
post4 = nn.Sequential()
post4:add(nn.SpatialFullConvolution(128,64,3,3,2,2,0,0,1,1))
post4:add(nn.SpatialFullConvolution(64,1,3,3,1,1))
post4:add(nn.ReLU())
post4:add(nn.SpatialFullConvolution(1,1,9,9,1,1))
post4:add(nn.ReLU())
post4:add(nn.Mul())
post:add(post4)
post5 = nn.Sequential()
post5:add(nn.SpatialFullConvolution(64,1,5,5,1,1))
post5:add(nn.ReLU())
post5:add(nn.Mul())
post:add(post5)

fullm:add(post)
fullm:add(nn.CAddTable())



fullm = fullm:cuda()

--initial branch
mlp = nn.ConcatTable()

cutorch.synchronize()

criterion = cudnn.SpatialCrossEntropyCriterion():cuda()
--criterion = nn.MSECriterion():cuda()
parameters,gradParameters = fullm:getParameters()

function train()
    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,input:size()[1] do
        -- create mini batch
        local inputs = torch.CudaTensor(1,1,500,500)
        inputs[1] = input[t]
        local targets = output[t]

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- evaluate function for complete mini batch
            local outputs = fullm:forward(inputs)
            local f = criterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            fullm:backward(inputs, df_do)

            -- return f and df/dX
            return f,gradParameters
        end


        -- Perform SGD step:
        sgdState = sgdState or {
            learningRate = 0.05,
            momentum = 0,
            learningRateDecay = 5e-7
        }
        optim.sgd(feval, parameters, sgdState)
    end

    -- time taken
    time = sys.clock() - time
    time = time / input:size()[1]
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- save/log current netA
    if epoch % 100 == 0 then
        print 'hi'
    end 

    -- next epoch
    epoch = epoch + 1
end
