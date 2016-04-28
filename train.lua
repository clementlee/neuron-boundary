require 'model'
require 'optim'
require 'xlua'
require 'CustomCriterion'

function trainload()
    if not loaded then
        stacks.load()
        loaded = true
    end
end 

trainload()
weights = torch.zeros(2)
weights[1] = 1/0.8
weights[2] = 1/0.2
criterion = cudnn.SpatialCrossEntropyCriterion(weights):cuda()
--criterion = nn.BCECriterionA():cuda()
parameters, gradParameters = model:getParameters()

optState = {
    learningRate = 1
}

batchsize = 1
--crossvalidratio = 0.05
--totalsamples = stacks.input:size(1)
--trainsize = math.floor((1-crossvalidratio)*totalsamples)
trainsize = stacks.input:size(1)

function train()
    trainload()
    model:training()
    epoch = epoch or 1
    local time = sys.clock()

    if epoch % 5 ==  0 then optState.learningRate = optState.learningRate/2 end



    print(string.format('epoch #%d, batchsize=%d, trainsize=%d', epoch, batchsize, trainsize))

    local indices = torch.randperm(trainsize):long():split(batchsize)
    -- remove last element so that all the batches have equal size
    indices[#indices] = nil

    local i = 0
    local totalc = 0
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

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

            local output = model:forward(inputs)
            local f = criterion:forward(output, targets)
            totalc = totalc + (f*batchsize / #indices)

            local df_do = criterion:backward(output, targets)
            model:backward(inputs, df_do)

            return f,gradParameters
        end

        optim.adam(feval, parameters, optState)
        i = i + 1
    end

    time = sys.clock() - time

    print('error: '..totalc)

    epoch = epoch + 1
end


function trainloop()
    while true do
        train()
        savemodel()
        if epoch % 5 == 0 then
            sampleout()
        end
    end
end

a = cudnn.SpatialSoftMax():cuda()
function sampleout()
    model:evaluate()
    for i = 1,30 do
        c = model:forward(stacks.s3[i]:view(1,1,500,500):cuda())
        c = a:forward(c)
        c = c[1][2]
        c = c[1]
        c:add(-1* c:min())
        if c:max() ~= 0 then
            c:div(c:max())
        end
        image.save('output/stack03/03_out'..('%.2d' % i)..'.png', c)
    end
    for i = 1,30 do
        c = model:forward(stacks.s5[i]:view(1,1,500,500):cuda())
        c = a:forward(c)
        c = c[1][2]
        c = c[1]
        c:add(-1* c:min())
        if c:max() ~= 0 then
            c:div(c:max())
        end
        image.save('output/stack05/05_out'..('%.2d' % i)..'.png', c)
    end
    model:training()
end
