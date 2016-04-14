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



    addLayers(1,64):add(nn.Dropout(0.3))
    addLayers(64,64)
    local pool1 = MaxPooling(2,2,2,2):ceil()
    model:add(pool1)
    addLayers(64,128):add(nn.Dropout(0.4))
    addLayers(128,128)
    local pool2 = MaxPooling(2,2,2,2):ceil()
    model:add(pool2)
    addLayers(128,256):add(nn.Dropout(0.4))
    addLayers(256,256):add(nn.Dropout(0.4))
    addLayers(256,256)
    local pool3 = MaxPooling(2,2,2,2):ceil()
    model:add(pool3)
    addLayers(256,512):add(nn.Dropout(0.4))
    addLayers(512,512):add(nn.Dropout(0.4))
    addLayers(512,512)
    local pool4 = MaxPooling(2,2,2,2):ceil()
    model:add(pool4)
    addLayers(512,512):add(nn.Dropout(0.4))
    addLayers(512,512):add(nn.Dropout(0.4))
    addLayers(512,512)
    local pool5 = MaxPooling(2,2,2,2):ceil()
    model:add(pool5)

    model:add(nn.SpatialConvolution(512,512,1,1))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(512,512,1,1))
    model:add(nn.ReLU(true))

    --decoder
    model:add(nn.SpatialMaxUnpooling(pool5))
    addFullLayers(512,256):add(nn.Dropout(0.4))
    model:add(nn.SpatialMaxUnpooling(pool4))
    addFullLayers(256,128):add(nn.Dropout(0.4))
    model:add(nn.SpatialMaxUnpooling(pool3))
    addFullLayers(128,64):add(nn.Dropout(0.4))
    model:add(nn.SpatialMaxUnpooling(pool2))
    addFullLayers(64,32):add(nn.Dropout(0.4))
    model:add(nn.SpatialMaxUnpooling(pool1))
    --no relu on last layer
    model:add(nn.SpatialFullConvolution(32,1,5,5,1,1,2,2))
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
