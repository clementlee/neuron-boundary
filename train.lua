require 'model'
require 'optim'
require 'CustomCriterion'

loaded = false
function trainload()
    if not loaded then
        stacks.load()
        loaded = true
    end
end 

trainload()

weights = torch.zeros(2)
weights[1] = 0.9
weights[2] = 0.1
--criterion = cudnn.SpatialCrossEntropyCriterion(weights):cuda()
criterion = nn.BCECriterionA():cuda()

--container to produce two classes
--mcont = nn.Sequential()
--mcont:add(model)
--tab = nn.ConcatTable()
--tab:add(nn.Identity())
--seq1 = nn.Sequential()
--seq1:add(nn.MulConstant(-1))
--seq1:add(nn.AddConstant(1))
--tab:add(seq1)
--mcont:add(tab)
--mcont:add(nn.JoinTable(2))
--
--mcont = mcont:cuda()

parameters, gradParameters = model:getParameters()

sgdState = {
    learningRate = 0.1,
    weightDecay = 0.00005,
    momentum = 0.9,
    learningRateDecay = 1e-7
}

confusion = optim.ConfusionMatrix(2)



function train()
    local net = model
    trainload()
    epoch = epoch or 1
    local time = sys.clock()

    --if epoch % 25 ==  0 then sgdState.learningRate = sgdState.learningRate/2 end

    local batchsize = 4

    local trainsize = 8--stacks.input:size(1)

    print('training epoch #' .. epoch..', batchsize = '..batchsize..' datasize = '..trainsize)
    --targets = torch.CudaTensor(batchsize)
    --inputs = torch.CudaTensor(batchsize)

    local indices = torch.randperm(trainsize):long():split(batchsize)
    -- remove last element so that all the batches have equal size
    indices[#indices] = nil

    local i = 0
    for t,v in ipairs(indices) do
        if i % 12 == 0 then 
            io.write('.')
            io.flush()
        end
        local inputs = stacks.input:index(1, v)
        local targets = stacks.output:index(1, v)
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

            --flato = output:reshape(batchsize, 2, 500*500)
            --flato = flato:transpose(2,3)
            --flato = flato:reshape(batchsize * 500*500, 2)
            --flatt = targets:reshape(batchsize * 500*500)
            --flatt:add(1)
            --
            if epoch % 10 == 0 then
                output = output:view(batchsize*500*500):add(1):round()
                targets = targets:view(batchsize*500*500):add(1)

                confusion:batchAdd(output, targets)
            end 
            --confusion:batchAdd(flato, flatt)

            return f,gradParameters
        end

        optim.sgd(feval, parameters, sgdState)

        --remove from GPU
        inputs = inputs:float()
        targets = targets:float()
        i = i + 1
    end
    io.write('\n')

    time = sys.clock() - time

    if epoch % 10 == 0 then 
        confusion:updateValids()
        print(confusion)
    end
    print('completed in ' .. (time*1000) .. 'ms')
    if epoch % 10 == 0 then 
        print('training accuracy: ' .. (confusion.totalValid*100))
    end

    confusion:zero()

    epoch = epoch + 1
end

while true do
    train()
    if epoch % 50 == 0 then
        savemodel()
    end
end
