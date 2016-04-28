require 'image'
require 'paths'

stacks = {}

stacks.path = 'stacks.t7'
stacks.s3 = torch.Tensor(30,1,500,500)
stacks.s5 = torch.Tensor(30,1,500,500)

stacks.input = torch.Tensor(1,1,500,500)
stacks.output = torch.Tensor(1,1,500,500)

function stacks.load(override)
    override = override or false
    if not paths.filep(stacks.path) or override then
        local augfactor = 2*2 --rotate, vflip, hflip

        local s4size = 30
        local s6size = 30
        local s2size = 30
        local s7size = 80
        local s7crops = 4  -- use 4 crops

        local total = augfactor * (s2size + s4size + s6size + (s7size * s7crops))

        local index = 1
        local perm = torch.randperm(total)

        stacks.input = torch.FloatTensor(total, 1, 500, 500)
        stacks.output = torch.FloatTensor(total, 500, 500)

        print('loading ' .. total .. ' samples')

        local function add_pair(t1, t2) 
            for rot = 1, 2 do
                for rot = 1, 2 do
                    --add using permutation matrix
                    local reali = perm[index]
                    index = index + 1
                    stacks.input[reali][1] = t1
                    stacks.output[reali] = t2

                    t1 = image.hflip(t1)
                    t2 = image.hflip(t2)
                end
                t1 = image.vflip(t1)
                t2 = image.vflip(t2)
            end
        end

        --load stack 2
        for x = 1, s2size do
            local loadi = x - 1
            loadi = "%.2d" % loadi
            local t1 = image.load('data/stack02/02_raw' .. loadi .. '.png')
            local t2 = image.load('data/stack02/02_lbl' .. loadi .. '.png')

            add_pair(t1, t2)
        end
        print('loaded stack 2')

        --load stack 4
        for x = 1, s4size do
            local loadi = x - 1
            loadi = "%.2d" % loadi
            local t1 = image.load('data/stack04/04_raw' .. loadi .. '.png')
            local t2 = image.load('data/stack04/04_lbl' .. loadi .. '.png')

            add_pair(t1, t2)
        end
        print('loaded stack 4')

        --load stack 6
        for x = 1, s6size do
            local loadi = x - 1
            loadi = "%.2d" % loadi
            local t1 = image.load('data/stack06/06_raw' .. loadi .. '.png')
            local t2 = image.load('data/stack06/06_lbl' .. loadi .. '.png')

            add_pair(t1, t2)
        end
        print('loaded stack 6')

        --load stack 7
        for x = 1, s7size do
            local loadi = x - 1
            loadi = "%.2d" % loadi
            local t1 = image.load('data/stack07/07_raw' .. loadi .. '.png')
            local t2 = image.load('data/stack07/07_lbl' .. loadi .. '.png')

            local crops = {'tl', 'tr', 'bl', 'br'}

            for _,c in ipairs(crops) do 
                local c1 = image.crop(t1, c, 500, 500)
                local c2 = image.crop(t2, c, 500, 500)
                add_pair(c1, c2)
            end
        end
        print('loaded stack 7')

        --load stack 3 and 5
        for x = 1, 30 do
            local loadi = x - 1
            loadi = "%.2d" % loadi
            local t1 = image.load('data/stack03/03_raw' .. loadi .. '.png')
            stacks.s3[x] = t1
            t1 = image.load('data/stack05/05_raw' .. loadi .. '.png')
            stacks.s5[x] = t1
        end
        print('loaded stack 3 and 5')
    else
        print('loading from disk')
        stacks = torch.load(stacks.path)
    end
end

function stacks.normalize()
    m = torch.mean(stacks.input)
    s = torch.std(stacks.input)

    stacks.input:add(-1 * m)
    stacks.input:div(s)

    m = torch.mean(stacks.s3)
    s = torch.std(stacks.s3)
    stacks.s3:add(-1 * m)
    stacks.s3:div(s)

    m = torch.mean(stacks.s5)
    s = torch.std(stacks.s5)
    stacks.s5:add(-1 * m)
    stacks.s5:div(s)
end

function stacks.save() 
    torch.save(stacks.path, stacks)
end
