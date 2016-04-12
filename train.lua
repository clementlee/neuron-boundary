require 'model'
require 'optim'

loaded = false
function trainload()
    if not loaded then
        stacks.load()

        model.load()
        loaded = true
    end
end 

trainload()

criterion = cudnn.SpatialCrossEntropyCriterion():cuda()

parameters, gradParameters = model.net:getParameters()

sgdState = {
    learningRate = 0.1,
    weightDecay = 0.00005,
    momentum = 0.9,
    learningRateDecay = 1e-7
}

--confusion = optim.ConfusionMatrix(2)


function train()
    local net = model.net
    trainload()
    epoch = epoch or 1
    local time = sys.clock()

    if epoch % 25 ==  0 then sgdState.learningRate = sgdState.learningRate/2 end

    print('training epoch #' .. epoch)

    local totalsize = stacks.input:size()[1]

    local batchsize = 8

    local trainsize = 960

    targets = torch.CudaTensor(batchsize)
    inputs = torch.CudaTensor(batchsize)

    local indices = torch.randperm(trainsize):long():split(batchsize)
    -- remove last element so that all the batches have equal size
    indices[#indices] = nil


    for t,v in ipairs(indices) do
        local inputs = stacks.input:index(1, v)
        local targets =  stacks.output:index(1, v)
        inputs = inputs:cuda()
        targets = targets:cuda()


        local feval = function(x)
            collectgarbage()

            if x ~= parameters then
                parameters:copy(x)
            end

            gradParameters:zero()

            local output = net:forward(inputs)
            local f = criterion:forward(output, targets)

            local df_do = criterion:backward(output, targets)
            net:backward(inputs, df_do)

            --confusion:batchAdd(outputs, targets)

            return f,gradParameters
        end

        optim.sgd(feval, parameters, sgdState)
    end

    time = sys.clock() - time

    print('completed in ' .. (time*1000) .. 'ms')
    --print('training accuracy: ' .. (confusion.totalValid*100))

    --confusion:zero()

    epoch = epoch + 1
end

while true do
    train()
    if epoch % 25 == 0 then
        model.save()
    end
end
