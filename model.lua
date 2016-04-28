require 'cutorch'
require 'cunn'
require 'cudnn'
require 'stackdata'

model = nn.Sequential()

local MaxPooling = cudnn.SpatialMaxPooling
if not paths.filep('model.t7') then
    local function addLayers(nin, nout) 
        model:add(cudnn.SpatialConvolution(nin, nout,3,3,1,1,1,1))
        model:add(cudnn.SpatialBatchNormalization(nout, 1e-3))
        model:add(cudnn.ReLU(true))
        return model
    end

    addLayers(1,32):add(nn.SpatialDropout(0.3))
    addLayers(32,32)
    local pool1 = MaxPooling(2,2,2,2):ceil()
    model:add(pool1)
    addLayers(32,64):add(nn.SpatialDropout(0.4))
    addLayers(64,64)
    local pool2 = MaxPooling(2,2,2,2):ceil()
    model:add(pool2)
    addLayers(64,128):add(nn.SpatialDropout(0.4))
    addLayers(128,128):add(nn.SpatialDropout(0.4))
    addLayers(128,128):add(nn.SpatialDropout())
    addLayers(128,64)
    model:add(nn.SpatialUpSamplingNearest(2))
    addLayers(64,64):add(nn.SpatialDropout())
    addLayers(64,64)
    model:add(nn.SpatialUpSamplingNearest(2))
    addLayers(64,32):add(nn.SpatialDropout())
    addLayers(32,32):add(nn.SpatialDropout())
    addLayers(32,2)

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
        init'cudnn.SpatialConvolution'
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
