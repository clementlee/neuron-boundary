require 'cutorch'
require 'cunn'
require 'cudnn'
require 'stackdata'


model = {}
model.path = 'model.t7'


function model.load(override)
    override = override or false
    if not paths.filep(model.path) or override then
        local net = nn.Sequential()

        local function addLayers(nin, nout) 
            net:add(cudnn.SpatialConvolution(nin, nout,3,3,1,1,1,1))
            net:add(nn.SpatialBatchNormalization(nout, 1e-3))
            net:add(nn.ReLU(true))
            return net
        end

        addLayers(1,64):add(nn.SpatialDropout())
        net:add(nn.SpatialMaxPooling(2,2,2,2))
        addLayers(64,64):add(nn.SpatialDropout())
        addLayers(64,64):add(nn.SpatialDropout())
        addLayers(64,64):add(nn.SpatialDropout())
        addLayers(64,64):add(nn.SpatialDropout())
        addLayers(64,64):add(nn.SpatialDropout())
        addLayers(64,64):add(nn.SpatialDropout())
        net:add(nn.SpatialFullConvolution(64,64,3,3,2,2,1,1,1,1))
        addLayers(64,1)


        local function MSRinit(net)
            local function init(name)
                for k,v in pairs(net:findModules(name)) do
                    print('hi')
                    local n = v.kW*v.kH*v.nOutputPlane
                    v.weight:normal(0,math.sqrt(2/n))
                    v.bias:zero()
                end
            end
            -- have to do for both backends
            init'nn.SpatialConvolution'
            init'nn.SpatialFullConvolution'
        end

        MSRinit(net)
        net = net:cuda()
        model.net = net

    else
        model = torch.load(model.path)
    end
end

function model.save()
    torch.save(model.path, model)
end
