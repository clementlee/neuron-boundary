require 'cutorch'
require 'cunn'
require 'stackdata'

--cutorch.setDevice(2)

model = nn.Sequential()

local MaxPooling = nn.SpatialMaxPooling
if not paths.filep('model.t7') then
    local function addLayers(nin, nout) 
        model:add(nn.SpatialConvolution(nin, nout,3,3,1,1,1,1))
        model:add(nn.SpatialBatchNormalization(nout, 1e-3))
        model:add(nn.ReLU(true))
        return model
    end

    local function addFullLayers(nin, nout) 
        model:add(nn.SpatialFullConvolution(nin, nout,5,5,1,1,2,2))
        model:add(nn.SpatialBatchNormalization(nout, 1e-3))
        model:add(nn.ReLU(true))
        return model
    end



    addLayers(1,8):add(nn.Dropout(0.3))
    addLayers(8,8)
    local pool1 = MaxPooling(2,2,2,2):ceil()
    model:add(pool1)
    addLayers(8,16):add(nn.Dropout(0.4))
    addLayers(16,16)
    local pool2 = MaxPooling(2,2,2,2):ceil()
    model:add(pool2)
    addLayers(16,32):add(nn.Dropout(0.4))
    addLayers(32,32):add(nn.Dropout(0.4))
    addLayers(32,32)
    local pool3 = MaxPooling(2,2,2,2):ceil()
    model:add(pool3)
    addLayers(32,64):add(nn.Dropout(0.4))
    addLayers(64,64):add(nn.Dropout(0.4))
    addLayers(64,64):add(nn.Dropout(0.4))
    --local pool4 = MaxPooling(2,2,2,2):ceil()
    --model:add(pool4)
    --addLayers(64,64):add(nn.Dropout(0.4))
    --addLayers(64,64):add(nn.Dropout(0.4))
    --addLayers(64,64)
    --local pool5 = MaxPooling(2,2,2,2):ceil()
    --model:add(pool5)

    model:add(nn.SpatialConvolution(64,64,1,1))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.4))
    model:add(nn.SpatialConvolution(64,64,1,1))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.4))

    --decoder
    --model:add(nn.SpatialMaxUnpooling(pool5))
    --addFullLayers(64,32):add(nn.Dropout(0.4))
    --model:add(nn.SpatialMaxUnpooling(pool4))
    addFullLayers(64,32):add(nn.Dropout(0.4))
    model:add(nn.SpatialMaxUnpooling(pool3))
    addFullLayers(32,16):add(nn.Dropout(0.4))
    model:add(nn.SpatialMaxUnpooling(pool2))
    addFullLayers(16,8):add(nn.Dropout(0.4))
    model:add(nn.SpatialMaxUnpooling(pool1))
    --no relu on last layer
    model:add(nn.SpatialFullConvolution(8,1,5,5,1,1,2,2))
    model:add(nn.SpatialBatchNormalization(1, 1e-3))

    --sigmoid
    model:add(nn.Sigmoid())


    local function MSRinit(net)
        local function init(name)
            for k,v in pairs(net:findModules(name)) do
                --print('hi')
                local n = v.kW*v.kH*v.nOutputPlane
                v.weight:normal(0,math.sqrt(2/n))
                v.bias:zero()
            end
        end
        -- have to do for both conv types
        init'nn.SpatialConvolution'
        init'nn.SpatialFullConvolution'
    end

    MSRinit(model)
    model = model:cuda()
else 
    model = torch.load('model.t7')
end

function savemodel()
    model:clearState()
    torch.save('model.t7', model)
end
