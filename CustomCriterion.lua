local BCECriterionA, parent = torch.class('nn.BCECriterionA', 'nn.Criterion')

local eps = 1e-12

local beta = 0.80

local temp = torch.FloatTensor()

function BCECriterionA:__init(weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
    if weights ~= nil then
        assert(weights:dim() == 1, "weights input should be 1-D Tensor")
        self.weights = weights
    end
end


function BCECriterionA:__len()
    if (self.weights) then
        return #self.weights
    else
        return 0
    end
end

function BCECriterionA:updateOutput(input, target)
    -- - log(input) * target - log(1 - input) * (1 - target)

    assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local weights = self.weights
    local output

    buffer:resizeAs(input)

    if weights ~= nil and target:dim() ~= 1 then
        weights = self.weights:view(1, target:size(2)):expandAs(target)
    end

    -- log(input) * target * beta
    buffer:add(input, eps):log()
    if weights ~= nil then buffer:cmul(weights) end
    --buffer:mul(beta)

    output = torch.dot(target, buffer) * beta

    -- log(1 - input) * (1 - target)
    buffer:mul(input, -1):add(1):add(eps):log()
    if weights ~= nil then buffer:cmul(weights) end
    --buffer:mul(1-beta)

    output = output + torch.sum(buffer) * (1-beta)
    output = output - torch.dot(target, buffer) * (1-beta)

    if self.sizeAverage then
        output = output / input:nElement()
    end

    self.output = - output

    return self.output
end

function BCECriterionA:updateGradInput(input, target)
    -- - (target - input) / ( input (1 - input) )
    -- The gradient is slightly incorrect:
    -- It should have be divided by (input + eps) (1 - input + eps)
    -- but it is divided by input (1 - input + eps) + eps
    -- This modification requires less memory to be computed.

    assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    --https://www.wolframalpha.com/input/?i=gradient+-z*log(x)*y+-+log(1-x)*(1-y)*(1-z)

    local buffer = self.buffer
    local weights = self.weights
    local gradInput = self.gradInput

    if weights ~= nil and target:dim() ~= 1 then
        weights = self.weights:view(1, target:size(2)):expandAs(target)
    end

    buffer:resizeAs(input)
    -- - x ( 1 + eps -x ) + eps
    buffer:add(input, -1):add(-eps):cmul(input):add(-eps)

    gradInput:resizeAs(input)
    -- y - x
    --
    -- x =  input
    -- z = beta
    -- y = target
    -- x (-2yz + y + z - 1) + yz
    --gradInput:add(target, -1, input)
    --
    gradInput:mul(target, 1-2*beta):add(beta-1):cmul(input)
    gradInput:add(beta, target)

    

    -- - (y - x) / ( x ( 1 + eps -x ) + eps )
    gradInput:cdiv(buffer)

    if weights ~= nil then
        gradInput:cmul(weights)
    end
    --gradInput:mul(10)

    if self.sizeAverage then
        gradInput:div(target:nElement())
    end

    return gradInput
end
