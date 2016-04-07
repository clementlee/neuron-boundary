require 'model2'
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

stacks.input = stacks.input:cuda()
stacks.output = stacks.output:cuda()

function train()
    local net = model.net
    trainload()
    epoch = epoch or 1
    local time = sys.clock()

    print('training epoch #' .. epoch)

    local totalsize = stacks.input:size()[1]

    local batchsize = 8
    for t = 1,totalsize,batchsize do

        top = math.min(t+batchsize - 1, totalsize)
        local testin = stacks.input[{{t,top}}]
        local testout = stacks.output[{{t,top}}]

        local feval = function(x)
            collectgarbage()

            if x ~= parameters then
                parameters:copy(x)
            end

            gradParameters:zero()

            local output = net:forward(testin)
            local f = criterion:forward(output, testout)

            local df_do = criterion:backward(output, testout)
            net:backward(testin, df_do)

            return f,gradParameters
        end

        sgdState = sgdState or {
            learningRate = 0.5,
            momentum = 0.9,
            learningRateDecay = 1e-6
        }

        optim.sgd(feval, parameters, sgdSate)
    end

    time = sys.clock() - time

    print('completed in ' .. (time*1000) .. 'ms')

    epoch = epoch + 1
end
