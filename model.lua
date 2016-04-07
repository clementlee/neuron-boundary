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
        net:add(cudnn.SpatialConvolution(1,64,3,3,1,1,1,1))
        net:add(nn.ReLU())
        net:add(nn.SpatialDropout())
        net:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))
        net:add(nn.ReLU())
        net:add(nn.SpatialDropout())
        net:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))
        net:add(nn.ReLU())
        net:add(nn.SpatialDropout())
        net:add(cudnn.SpatialConvolution(64,1,3,3,1,1,1,1))
        net:add(nn.ReLU())
        net = net:cuda()
        model.net = net
    else
        model = torch.load(model.path)
    end
end

function model.save()
    torch.save(model.path. model)
end
